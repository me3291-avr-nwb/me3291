import sys
import os
import getopt
import numpy as np
import matplotlib.pyplot as plt
from part_one import qn1a
from part_two import qn1b
from part_three import qn2

options = "q:"
long_options = ["qn"]  # Long options


def main(argv):
    np.set_printoptions(precision=3)
    try:
        opts, args = getopt.getopt(argv, options, long_options)

        for opt, arg in opts:
            if opt in ("-q", "--qn"):
                choices = {'1a': qn1a, '1b': qn1b, '2': qn2}
                choices[arg]()

        if len(argv) == 0:
            raise getopt.error("No arguments given")
    except getopt.error as err:
        print(str(err))
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
