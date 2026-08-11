import numpy as np


def rotation_matrix(omega, phi, kappa):
    R = np.array([
        [np.cos(phi) * np.cos(kappa), -np.cos(phi) * np.sin(kappa), np.sin(phi)],
        [np.cos(omega) * np.sin(kappa) + np.sin(omega) * np.sin(phi) * np.cos(kappa),
         np.cos(omega) * np.cos(kappa) - np.sin(omega) * np.sin(phi) * np.sin(kappa),
         -np.sin(omega) * np.cos(phi)],
        [np.sin(omega) * np.sin(kappa) - np.cos(omega) * np.sin(phi) * np.cos(kappa),
         np.sin(omega) * np.cos(kappa) + np.cos(omega) * np.sin(phi) * np.sin(kappa),
         np.cos(omega) * np.cos(phi)]
    ])
    return R


def collinearity_equations(X, Y, Z, x0, y0, f, R, X0, Y0, Z0):
    diff = np.array([X - X0, Y - Y0, Z - Z0])
    x_calc = x0 - f * (R[0, :] @ diff) / (R[2, :] @ diff)
    y_calc = y0 - f * (R[1, :] @ diff) / (R[2, :] @ diff)
    return x_calc, y_calc


def bundle_adjustment(observations1, observations2, control_points, initial_params1, initial_params2, max_iters=10,
                      threshold=1e-6):
    # Initialize parameters
    params1 = initial_params1
    params2 = initial_params2

    for iteration in range(max_iters):
        R1 = rotation_matrix(params1["omega"], params1["phi"], params1["kappa"])
        R2 = rotation_matrix(params2["omega"], params2["phi"], params2["kappa"])

        residuals = []
        A_matrix = []

        for obs in observations1 + observations2:
            x_obs, y_obs, point_id = obs["x"], obs["y"], obs["point_id"]
            X, Y, Z = control_points[point_id]

            if obs["image"] == 1:
                x_calc, y_calc = collinearity_equations(X, Y, Z, params1["x0"], params1["y0"], params1["f"], R1,
                                                        params1["X0"], params1["Y0"], params1["Z0"])
            else:
                x_calc, y_calc = collinearity_equations(X, Y, Z, params2["x0"], params2["y0"], params2["f"], R2,
                                                        params2["X0"], params2["Y0"], params2["Z0"])

            residuals.extend([x_obs - x_calc, y_obs - y_calc])
            # Add linearized partial derivatives to A matrix (skip detail here for brevity)

        residuals = np.array(residuals)

        # Least squares adjustment to solve for parameter updates

        if np.linalg.norm(residuals) < threshold:
            break

    return {"params_image1": params1, "params_image2": params2}


# Initial parameters
initial_params1 = {"x0": 0, "y0": 0, "f": 100.5, "X0": 500, "Y0": 500, "Z0": 300, "omega": 0, "phi": 0, "kappa": 0}
initial_params2 = {"x0": 0, "y0": 0, "f": 100.5, "X0": 600, "Y0": 600, "Z0": 300, "omega": 0, "phi": 0, "kappa": 0}

# Observations from both images
observations1 = [{"x": 1024.3, "y": 768.5, "point_id": 1, "image": 1},
                 {"x": 1040.2, "y": 780.9, "point_id": 2, "image": 1}]
observations2 = [{"x": 2048.6, "y": 1536.9, "point_id": 1, "image": 2},
                 {"x": 2060.1, "y": 1550.7, "point_id": 2, "image": 2}]

control_points = {1: (1000, 2000, 300), 2: (1100, 2100, 350)}

result = bundle_adjustment(observations1, observations2, control_points, initial_params1, initial_params2)
print(result)
