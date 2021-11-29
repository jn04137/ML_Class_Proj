# AUTHOR: Jonathan Nguyen & Austin Porter
import numpy as np

def main():
    heist_data = np.genfromtxt('adult.data', delimiter=',', skip_header=1)
    print(heist_data)
    print("Hello world!")


if __name__ == "__main__":
    main()

