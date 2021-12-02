import numpy as np


def main():
    census_data = np.genfromtxt('adult.data', delimiter=',', skip_header=1)
    print(census_data)


if __name__ == "__main__":
    main()
