import matplotlib.pyplot as plt
import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
       'datapoints', help='speficy the number of data points to generate per class', type=int)
    args = parser.parse_args()
    gen(4, args.datapoints)
