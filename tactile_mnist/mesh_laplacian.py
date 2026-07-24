from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import threadpoolctl
import trimesh
from transformation import Transformation


def cotangent_laplacian(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """
    Compute the cotangent Laplace-Beltrami stiffness matrix and the barycentric lumped mass matrix of a triangle mesh.

    The returned stiffness matrix L is symmetric positive semi-definite (L @ const = 0) and the mass matrix is returned
    as a vector of per-vertex masses (one third of the total area of the incident faces).

    :param vertices: Vertex positions of shape (N, 3).
    :param faces: Triangle indices of shape (F, 3).
    :return: Tuple of the sparse stiffness matrix L of shape (N, N) and the per-vertex mass vector of shape (N,).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(vertices)

    # Skip degenerate faces as they yield unbounded cotangent weights
    double_area = np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=-1,
    )
    max_double_area = np.max(double_area, initial=0.0)
    valid = double_area > max(1e-12, 1e-8 * max_double_area)
    faces = faces[valid]
    double_area = double_area[valid]

    rows = []
    cols = []
    weights = []
    for c in range(3):
        i, j, k = faces[:, c], faces[:, (c + 1) % 3], faces[:, (c + 2) % 3]
        e_ij = vertices[j] - vertices[i]
        e_ik = vertices[k] - vertices[i]
        # Half cotangent of the angle at vertex i, weighting the opposite edge (j, k)
        w = 0.5 * (e_ij * e_ik).sum(-1) / double_area
        rows.extend([j, k, j, k])
        cols.extend([k, j, j, k])
        weights.extend([-w, -w, w, w])

    stiffness = scipy.sparse.coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()

    vertex_mass = np.zeros(n, dtype=np.float64)
    np.add.at(vertex_mass, faces.reshape(-1), np.repeat(double_area / 6, 3))

    return stiffness, vertex_mass


@dataclass(frozen=True)
class SpectralShapeRepresentation:
    """
    Truncated spectral (Laplace-Beltrami eigenbasis) representation of a triangle mesh.

    The eigenvectors form an orthonormal basis w.r.t. the mass matrix inner product, so the coefficients of the vertex
    positions are given by C = eigenvectors^T diag(vertex_mass) V and the (smoothed) reconstruction of the shape is
    V ~ eigenvectors @ C.
    """

    # Eigenvalues of the Laplace-Beltrami operator in ascending order, zero-padded to num_coefficients entries.
    eigenvalues: np.ndarray
    # Corresponding eigenvectors of shape (num_vertices, num_coefficients), zero-padded like the eigenvalues.
    eigenvectors: np.ndarray
    # Lumped per-vertex masses of shape (num_vertices,).
    vertex_mass: np.ndarray
    # Spectral coefficients of the vertex positions in the mesh's model frame, of shape (num_coefficients, 3).
    coefficients: np.ndarray
    # Projection of the constant function 1 onto the eigenbasis, of shape (num_coefficients,).
    constant_projection: np.ndarray

    @property
    def num_coefficients(self) -> int:
        return self.eigenvectors.shape[1]

    @property
    def total_mass(self) -> float:
        return float(self.vertex_mass.sum())

    @property
    def rms_radius(self) -> float:
        """
        Mass-weighted RMS distance of the truncated reconstruction from the mass centroid; a measure of object scale.
        """
        # Constant modes (eigenvalue ~ 0) carry the centroid, all other modes the actual shape
        non_constant = self.eigenvalues > 1e-8 * np.max(self.eigenvalues, initial=0.0)
        return float(
            np.linalg.norm(self.coefficients[non_constant]) / np.sqrt(self.total_mass)
        )

    def transform_coefficients(self, pose: Transformation) -> np.ndarray:
        """
        Express the spectral coefficients of this mesh after transforming it by the given rigid pose.

        Since V' = V @ R^T + t^T, the coefficients become C' = C @ R^T + constant_projection t^T.

        :param pose: Single rigid transformation to apply to the mesh.
        :return: Transformed spectral coefficients of shape (num_coefficients, 3).
        """
        rotation_matrix = pose.rotation.as_matrix()
        return self.coefficients @ rotation_matrix.T + np.outer(
            self.constant_projection, pose.translation
        )

    def reconstruct(self, coefficients: np.ndarray | None = None) -> np.ndarray:
        """
        Reconstruct per-vertex positions from spectral coefficients.

        :param coefficients: Spectral coefficients of shape (num_coefficients, 3). Defaults to this representation's
                             own (model frame) coefficients, yielding a smoothed version of the original mesh.
        :return: Reconstructed vertex positions of shape (num_vertices, 3).
        """
        if coefficients is None:
            coefficients = self.coefficients
        return self.eigenvectors @ coefficients


def compute_spectral_representation(
    mesh: trimesh.Trimesh, num_coefficients: int
) -> SpectralShapeRepresentation:
    """
    Compute the truncated Laplace-Beltrami spectral representation of a mesh.

    The eigenvector signs are fixed by requiring the entry of largest magnitude to be positive, making the
    representation deterministic for a given mesh (up to eigenvalue multiplicities).

    :param mesh: Mesh to compute the representation for.
    :param num_coefficients: Number of spectral coefficients (eigenpairs) to keep. If the mesh has fewer vertices than
                             that, the representation is zero-padded.
    :return: The spectral representation of the mesh.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n = len(vertices)

    stiffness, vertex_mass = cotangent_laplacian(vertices, faces)

    # Restrict the eigenproblem to vertices that carry mass (i.e. are referenced by at least one non-degenerate face),
    # as zero-mass vertices would make the mass matrix singular.
    active = vertex_mass > 0
    n_active = int(np.sum(active))
    if n_active == 0:
        raise ValueError("Mesh does not contain any non-degenerate faces.")
    stiffness_active = stiffness[active][:, active]
    mass_active = vertex_mass[active]

    k = min(num_coefficients, n_active)
    # Restrict the eigensolver to a single BLAS thread, as its internal dense operations are too small to benefit from
    # multithreading, which just adds overhead here.
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        if k >= n_active - 1:
            eigenvalues, eigenvectors_active = scipy.linalg.eigh(
                stiffness_active.toarray(), np.diag(mass_active)
            )
            eigenvalues = eigenvalues[:k]
            eigenvectors_active = eigenvectors_active[:, :k]
        else:
            mass_matrix = scipy.sparse.diags(mass_active).tocsc()
            # Use a fixed starting vector to make the decomposition deterministic, as ARPACK would otherwise return
            # arbitrarily mixed eigenvectors for degenerate (repeated) eigenvalues of symmetric meshes.
            v0 = np.random.default_rng(0).uniform(-1, 1, size=n_active)
            try:
                # Shift-invert around a small negative sigma, as the stiffness matrix itself is singular
                eigenvalues, eigenvectors_active = scipy.sparse.linalg.eigsh(
                    stiffness_active, k=k, M=mass_matrix, sigma=-1e-2, v0=v0
                )
            except (
                scipy.sparse.linalg.ArpackNoConvergence,
                RuntimeError,
            ):
                eigenvalues, eigenvectors_active = scipy.linalg.eigh(
                    stiffness_active.toarray(),
                    np.diag(mass_active),
                    subset_by_index=[0, k - 1],
                )
            order = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[order]
            eigenvectors_active = eigenvectors_active[:, order]

    # Numerical noise can make the (theoretically non-negative) eigenvalues slightly negative
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Fix the sign ambiguity of the eigenvectors deterministically
    sign_idx = np.argmax(np.abs(eigenvectors_active), axis=0)
    signs = np.sign(eigenvectors_active[sign_idx, np.arange(k)])
    signs[signs == 0] = 1.0
    eigenvectors_active = eigenvectors_active * signs

    eigenvectors = np.zeros((n, num_coefficients), dtype=np.float64)
    eigenvectors[active, :k] = eigenvectors_active
    eigenvalues_padded = np.zeros(num_coefficients, dtype=np.float64)
    eigenvalues_padded[:k] = eigenvalues

    mass_weighted_basis = eigenvectors * vertex_mass[:, None]
    coefficients = mass_weighted_basis.T @ vertices
    constant_projection = mass_weighted_basis.sum(axis=0)

    return SpectralShapeRepresentation(
        eigenvalues=eigenvalues_padded,
        eigenvectors=eigenvectors,
        vertex_mass=vertex_mass,
        coefficients=coefficients,
        constant_projection=constant_projection,
    )
