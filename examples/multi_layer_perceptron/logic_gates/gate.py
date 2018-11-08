import argparse
import sys
newline = "\n"
def file_is_present(file_handle):
    return file_handle != None

def make_inputs(gate, file_handle, num_points)  :
    if file_is_present(file_handle):
        file_handle.write(gate + " gate inputs" + newline)
        for i in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    file_handle.write("0" + "0" + newline)
                elif i == 1:
                    file_handle.write("0" + "1" + newline)
                elif i == 2:
                    file_handle.write("1" + "0" + newline)
                elif i == 3:
                    file_handle.write("1" + "1" + newline)
    else :
        print ("no file")
        print (gate + 'gate labels')
        for i  in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    print('0 0')
                elif i == 1:
                    print('0 1')
                elif i == 2:
                    print('1 0')
                elif i == 3:
                    print('1 0')

def make_or(file_handle, num_points)   :
    gate = 'and'
    if file_is_present(file_handle):
        file_handle.write(gate + ' gate labels')
    else :
        print ("no file")
        print (gate)

'''
def make_nand() :
def make_nor()  :
def make_not()  :
'''
def make_data(parser):
    args = parser.parse_args()
    if args.number == None:
        sys.exit()

    if args.gate == 'and'  :
        make_inputs(args.gate, args.input, args.number)
    elif args.gate == 'or'   :
        make_or(args.label, args.number)
    elif args.gate == 'nand' :
        print (args.gate)
    elif args.gate == 'nor'  :
        print (args.gate)

    elif args.gate == 'xor' :
        print (args.gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gate',   type=str, choices=['or', 'and', 'nor', 'nand', 'xor'], help='the logic to choose from')
    parser.add_argument('-i', '--input',  type=argparse.FileType('w'), help='inputs output file')
    parser.add_argument('-t', '--target', type=argparse.FileType('w'), help='targets output file')
    parser.add_argument('-n', '--number', type=int, help='number of targets and labels for each class')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    make_data(parser)
    

