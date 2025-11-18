from helper_v_point import diamond_accumulator

def main():
    w = 150
    h = 100
    output_path = "export"
    all_lines = [[0, 20, 10, 30], [20, 40, 30, 50], [40, 60, 50, 70], [60, 80, 70, 90]]

    diamond_accumulator(all_lines, w, h, output_path, 1)

if __name__ == "__main__":
    main()