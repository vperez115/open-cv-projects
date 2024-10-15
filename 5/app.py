import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from tkinter import Tk, filedialog

def upload_csv_file():
    """Open a file dialog to upload a CSV file."""
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

# Upload the CSV file
csv_path = upload_csv_file()

if csv_path:
    # Load the uploaded CSV file into a DataFrame
    d0 = pd.read_csv(csv_path)
    print(d0.head(5))  # Print first five rows

    # Save the labels into a variable l.
    l = d0['label']

    # Drop the label feature and store the pixel data in d.
    d = d0.drop("label", axis=1)
    print(d.shape)
    print(l.shape)

    def display():
        """Display or plot a number from the dataset."""
        plt.figure(figsize=(7, 7))
        idx = 1

        grid_data = d.iloc[idx].to_numpy().reshape(28, 28)  # Reshape from 1D to 2D pixel array
        plt.imshow(grid_data, interpolation="none", cmap="gray")
        plt.show()

        print(l[idx])

    def label():
        """Perform analysis on a subset of the data."""
        # Pick first 15K data-points to work on for time-efficiency.
        labels = l.head(15000)
        data = d.head(15000)

        print("The shape of sample data =", data.shape)

        # Data-preprocessing: Standardizing the data
        standardized_data = StandardScaler().fit_transform(data)
        print(standardized_data.shape)

        # Find the covariance matrix: A^T * A
        sample_data = standardized_data

        # Matrix multiplication using numpy
        covar_matrix = np.matmul(sample_data.T, sample_data)
        print("The shape of variance matrix =", covar_matrix.shape)

        # Find the top two eigenvalues and corresponding eigenvectors
        values, vectors = eigh(covar_matrix, subset_by_index=[782, 783])

        print("Shape of eigen vectors =", vectors.shape)

        # Convert the eigenvectors into (2, d) shape for easier computations
        vectors = vectors.T
        print("Updated shape of eigen vectors =", vectors.shape)

        # Here, vectors[1] is the eigenvector corresponding to the 1st principal eigenvalue,
        # and vectors[0] is the eigenvector corresponding to the 2nd principal eigenvalue.

    # Call the label function to perform the analysis
    label()

else:
    print("No file selected.")
