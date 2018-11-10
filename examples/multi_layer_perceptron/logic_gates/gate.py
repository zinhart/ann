import argparse
import sys
import struct
newline = "\n"
def file_is_present(file_handle):
    return file_handle != None

def make_inputs(gate, file_handle, num_points) :
    if file_is_present(file_handle):
        file_handle.write(struct.pack('7B', 122, 105, 110, 104, 97, 114, 116))  # zinhart
        file_handle.write(struct.pack('I', num_points))
        for i in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    file_handle.write(struct.pack('B', 0))
                    file_handle.write(struct.pack('B', 0))
                elif i == 1:
                    file_handle.write(struct.pack('B', 0))
                    file_handle.write(struct.pack('B', 1))
                elif i == 2:
                    file_handle.write(struct.pack('B', 1))
                    file_handle.write(struct.pack('B', 0))
                elif i == 3:
                    file_handle.write(struct.pack('B', 1))
                    file_handle.write(struct.pack('B', 1))
    else :
        print (gate + " gate labels")
        for i  in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    print("0" + " 0")
                elif i == 1:
                    print("0" + " 1")
                elif i == 2:
                    print("1" + " 0")
                elif i == 3:
                    print("1" + " 1")

def make_and_gate_targets(gate, file_handle, num_points) :
    if file_is_present(file_handle):
        file_handle.write(struct.pack('7B', 122, 105, 110, 104, 97, 114, 116))  # zinhart
        file_handle.write(struct.pack('I', num_points))
        for i in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    file_handle.write(struct.pack('B', 0))
                elif i == 1:
                    file_handle.write(struct.pack('B', 0))
                elif i == 2:
                    file_handle.write(struct.pack('B', 0))
                elif i == 3:
                    file_handle.write(struct.pack('B', 1))
    else :
        print (gate + " gate targets")
        for i  in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    print("0")
                elif i == 1:
                    print("0")
                elif i == 2:
                    print("0")
                elif i == 3:
                    print("1")

def make_or_gate_targets(gate, file_handle, num_points) :
    if file_is_present(file_handle):
        file_handle.write(struct.pack('7B', 122, 105, 110, 104, 97, 114, 116))  # zinhart
        file_handle.write(struct.pack('I', num_points))
        for i in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    file_handle.write(struct.pack('B', 0))
                elif i == 1:
                    file_handle.write(struct.pack('B', 1))
                elif i == 2:
                    file_handle.write(struct.pack('B', 1))
                elif i == 3:
                    file_handle.write(struct.pack('B', 1))
    else :
        print (gate + " gate targets")
        for i  in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    print("0")
                elif i == 1:
                    print("1")
                elif i == 2:
                    print("1")
                elif i == 3:
                    print("1")

def make_nand_gate_targets(gate, file_handle, num_points) :
    if file_is_present(file_handle):
        file_handle.write(struct.pack('7B', 122, 105, 110, 104, 97, 114, 116))  # zinhart
        file_handle.write(struct.pack('I', num_points))
        for i in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    file_handle.write(struct.pack('B', 1))
                elif i == 1:
                    file_handle.write(struct.pack('B', 1))
                elif i == 2:
                    file_handle.write(struct.pack('B', 1))
                elif i == 3:
                    file_handle.write(struct.pack('B', 0))
    else :
        print (gate + " gate targets")
        for i  in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    print("1")
                elif i == 1:
                    print("1")
                elif i == 2:
                    print("1")
                elif i == 3:
                    print("0")

def make_nor_gate_targets(gate, file_handle, num_points) :
    if file_is_present(file_handle):
        file_handle.write(struct.pack('7B', 122, 105, 110, 104, 97, 114, 116))  # zinhart
        file_handle.write(struct.pack('I', num_points))
        for i in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    file_handle.write(struct.pack('B', 1))
                elif i == 1:
                    file_handle.write(struct.pack('B', 0))
                elif i == 2:
                    file_handle.write(struct.pack('B', 0))
                elif i == 3:
                    file_handle.write(struct.pack('B', 0))
    else :
        print (gate + " gate targets")
        for i  in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    print("1")
                elif i == 1:
                    print("0")
                elif i == 2:
                    print("0")
                elif i == 3:
                    print("0")

def make_xor_gate_targets(gate, file_handle, num_points) :
    if file_is_present(file_handle):
        file_handle.write(struct.pack('7B', 122, 105, 110, 104, 97, 114, 116))  # zinhart
        file_handle.write(struct.pack('I', num_points))
        for i in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    file_handle.write(struct.pack('B', 0))
                elif i == 1:
                    file_handle.write(struct.pack('B', 1))
                elif i == 2:
                    file_handle.write(struct.pack('B', 1))
                elif i == 3:
                    file_handle.write(struct.pack('B', 0))
    else :
        print (gate + " gate targets")
        for i  in range(0,4):
            for j in range(0, num_points):
                if i == 0:
                    print("0")
                elif i == 1:
                    print("1")
                elif i == 2:
                    print("1")
                elif i == 3:
                    print("0")


def make_data(parser):
    args = parser.parse_args()
    if args.number == None:
        print("must specify -n argument")
        sys.exit()
        
    make_inputs(args.gate, args.input, args.number)

    if args.gate == 'and'  :
       # print()
        make_and_gate_targets(args.gate, args.target, args.number)
    elif args.gate == 'or'   :
        make_or_gate_targets(args.gate, args.input, args.number)
    elif args.gate == 'nand' :
        make_nand_gate_targets(args.gate, args.input, args.number)
    elif args.gate == 'nor'  :
        make_nor_gate_targets(args.gate, args.input, args.number)
    elif args.gate == 'xor' :
        make_xor_gate_targets(args.gate, args.input, args.number)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gate',   type=str, choices=['and','or','nand', 'nor', 'xor'], help='the logic to choose from')
    parser.add_argument('-i', '--input',  type=argparse.FileType('wb'), help='inputs output file')
    parser.add_argument('-t', '--target', type=argparse.FileType('wb'), help='targets output file')
    parser.add_argument('-n', '--number', type=int, help='number of target and label pairs for each class')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    make_data(parser)
    

