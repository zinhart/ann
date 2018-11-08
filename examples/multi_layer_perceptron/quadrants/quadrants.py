import matplotlib.pyplot as plt
import argparse
import sys
import random

def gen(n_classes, data_points_per_class):
    fig    = plt.figure()
    ax     = fig.add_subplot(1, 1, 1)
    data   = open("data-quadrants", "w")
    labels = open("labels-quadrants", "w")

    for x in range(n_classes):
        for y in range(data_points_per_class):
            if  x == 0:
                x1 = random.uniform(0, 1)
                x2 = random.uniform(0, 1)
                labels.write("1,0,0,0\n")
                data.write(str(x1) + "," + str(x2)+ "\n")
                ax.plot(x1,x2, "ro")
            elif x == 1:
                x1 = random.uniform(0, -1)
                x2 = random.uniform(0, 1)
                labels.write("0,1,0,0\n")
                data.write(str(x1) + "," + str(x2)+ "\n")
                ax.plot(x1,x2, "go")
            elif x == 2:
                x1 = random.uniform(0, -1)
                x2 = random.uniform(0, -1)
                labels.write("0,0,1,0\n")
                data.write(str(x1) + "," + str(x2)+ "\n")
                ax.plot(x1,x2, "bo")
            elif x == 3:
                x1 = random.uniform(0, 1)
                x2 = random.uniform(0, -1)
                labels.write("0,0,0,1\n")
                data.write(str(x1) + "," + str(x2)+ "\n")
                ax.plot(x1,x2, "yo")

    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.show()


newline = "\n"
def file_is_present(file_handle):
    return file_handle != None

def get_figure():
    return plt.figure()

def get_subplot(figure, x1, y, x2):
    return figure.add_subplot(x1, y, x2)

def subplot_setting(subplot):
    if subplot != None:
        subplot.spines['left'].set_position('center')
        subplot.spines['right'].set_color('none')
        subplot.spines['bottom'].set_position('center')
        subplot.spines['top'].set_color('none')
        subplot.spines['left'].set_smart_bounds(True)
        subplot.spines['bottom'].set_smart_bounds(True)
        subplot.xaxis.set_ticks_position('bottom')
        subplot.yaxis.set_ticks_position('left')

def subplot_plot_point(subplot, x1, x2, color): 
    if subplot != None:
        subplot.plot(x1,x2, color)

def show_plot():
    plt.show()

def make_inputs(quadrant, file_handle, num_points, subplot):
    if file_is_present(file_handle):
        file_handle.write("quarant " + quadrant + " inputs" + newline)
        for i in range(0, num_points):
            if quadrant == "1":
                x1 = random.uniform(0, 1)
                x2 = random.uniform(0, 1)
                subplot_plot_point(subplot, x1,x2, "ro")
                file_handle.write(str(x1)+ "," + str(x2) + newline)
            elif quadrant == "2":
                x1 = random.uniform(0, -1)
                x2 = random.uniform(0, 1)
                subplot_plot_point(subplot, x1,x2, "go")
                file_handle.write(str(x1) + "," + str(x2) + newline)
            elif quadrant == "3":
                x1 = random.uniform(0, -1)
                x2 = random.uniform(0, -1)
                subplot_plot_point(subplot, x1,x2, "yo")
                file_handle.write(str(x1) + "," + str(x2) + newline)
            elif quadrant == "4":
                x1 = random.uniform(0, 1)
                x2 = random.uniform(0, -1)
                subplot_plot_point(subplot, x1,x2, "bo")
                file_handle.write(str(x1) + "," + str(x2) + newline)
    else:
        print("quadrant " + quadrant + " inputs")
        for i in range(0, num_points):
            if quadrant == "1":
                x1 = random.uniform(0, 1)
                x2 = random.uniform(0, 1)
                subplot_plot_point(subplot, x1,x2, "ro")
                print(str(x1)+ "," + str(x2))
            elif quadrant == "2":
                x1 = random.uniform(0, -1)
                x2 = random.uniform(0, 1)
                subplot_plot_point(subplot, x1,x2, "go")
                print(str(x1) + "," + str(x2))
            elif quadrant == "3":
                x1 = random.uniform(0, -1)
                x2 = random.uniform(0, -1)
                subplot_plot_point(subplot, x1,x2, "yo")
                print(str(x1) + "," + str(x2))
            elif quadrant == "4":
                x1 = random.uniform(0, 1)
                x2 = random.uniform(0, -1)
                subplot_plot_point(subplot, x1,x2, "bo")
                print(str(x1) + "," + str(x2))

def make_targets(quadrant, file_handle, num_points):
    if file_is_present(file_handle):
        file_handle.write("quarant " + quadrant + " targets" + newline)
        for i in range(0, num_points):
            if  quadrant == "1":
                file_handle.write(str("1,0,0,0") + newline)
            elif quadrant == "2":
                file_handle.write(str("0,1,0,0") + newline)
            elif quadrant == "3":
                file_handle.write(str("0,0,1,0") + newline)
            elif quadrant == "4":
                file_handle.write(str("0,0,0,1") + newline)
    else:
        print("quadrant " + quadrant + " targets")
        for i in range(0, num_points):
            if quadrant == "1":
                print(str("1,0,0,0"))
            elif quadrant == "2":
                print(str("0,1,0,0"))
            elif quadrant == "3":
                print(str("0,0,1,0"))
            elif quadrant == "4":
                print(str("0,0,0,1"))

def make_data(parser):
    args = parser.parse_args()
    if args.number == None:
        print("must specify -n argument")
        sys.exit()
    
    subplot = None
    if args.visualize == "y":
        subplot = get_subplot(get_figure(), 1,1,1)
    
#    make_inputs(args.quadrant, args.input, args.number, subplot)
#    make_targets(args.quadrant, args.input, args.number)

    for i in range(1, 5):
        if i == 1:
            make_inputs("1", args.input, args.number, subplot)
            make_targets("1", args.input, args.number)
        elif i == 2:
            make_inputs("2", args.input, args.number, subplot)
            make_targets("2", args.input, args.number)
        elif i == 3:
            make_inputs("3", args.input, args.number, subplot)
            make_targets("3", args.input, args.number)
        elif i == 4:
            make_inputs("4", args.input, args.number, subplot)
            make_targets("4", args.input, args.number)
    subplot_setting(subplot)
    show_plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quadrant',   type=str, choices=["1", '2', '3', '4'], help='the quadrant to choose from')
    parser.add_argument('-i', '--input',  type=argparse.FileType('w'), help='inputs output file')
    parser.add_argument('-t', '--target', type=argparse.FileType('w'), help='targets output file')
    parser.add_argument('-v', '--visualize',   type=str, choices=["y"], help='see points graphically or not')
    parser.add_argument('-n', '--number', type=int, help='number of target and label pairs for each class')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    make_data(parser)
  #  gen(4, args.datapoints)
