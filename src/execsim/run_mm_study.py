import os
from .fig_midprice_twap import main as f1
from .fig_ac_inventory import main as f2
from .fig_is_benchmark import main as f3

def main():
    os.makedirs("results", exist_ok=True)
    f1()
    f2()
    f3()

if __name__=="__main__":
    main()
