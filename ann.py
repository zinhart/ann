import os, sys, subprocess, docker, re, json
client = docker.from_env()
header = "\
 .----------------. .----------------. .-----------------..----------------. .----------------. .----------------. .----------------.\n\
| .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. |\n\
| |   ________   | | |     _____    | | | ____  _____  | | |  ____  ____  | | |      __      | | |  _______     | | |  _________   | |\n\
| |  |  __   _|  | | |    |_   _|   | | ||_   \|_   _| | | | |_   ||   _| | | |     /  \     | | | |_   __ \    | | | |  _   _  |  | |\n\
| |  |_/  / /    | | |      | |     | | |  |   \ | |   | | |   | |__| |   | | |    / /\ \    | | |   | |__) |   | | | |_/ | | \_|  | |\n\
| |     .'.' _   | | |      | |     | | |  | |\ \| |   | | |   |  __  |   | | |   / ____ \   | | |   |  __ /    | | |     | |      | |\n\
| |   _/ /__/ |  | | |     _| |_    | | | _| |_\   |_  | | |  _| |  | |_  | | | _/ /    \ \_ | | |  _| |  \ \_  | | |    _| |_     | |\n\
| |  |________|  | | |    |_____|   | | ||_____|\____| | | | |____||____| | | ||____|  |____|| | | |____| |___| | | |   |_____|    | |\n\
| |              | | |              | | |              | | |              | | |              | | |              | | |              | |\n\
| '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' |\n\
 '----------------' '----------------' '----------------' '----------------' '----------------' '----------------' '----------------'\n\
 .----------------. .----------------. .----------------. .----------------. .----------------. .----------------. .----------------. .----------------. .----------------. .----------------.\n\
| .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. |\n\
| |      __      | | |  _______     | | |  _________   | | |     _____    | | |  _________   | | |     _____    | | |     ______   | | |     _____    | | |      __      | | |   _____      | |\n\
| |     /  \     | | | |_   __ \    | | | |  _   _  |  | | |    |_   _|   | | | |_   ___  |  | | |    |_   _|   | | |   .' ___  |  | | |    |_   _|   | | |     /  \     | | |  |_   _|     | |\n\
| |    / /\ \    | | |   | |__) |   | | | |_/ | | \_|  | | |      | |     | | |   | |_  \_|  | | |      | |     | | |  / .'   \_|  | | |      | |     | | |    / /\ \    | | |    | |       | |\n\
| |   / ____ \   | | |   |  __ /    | | |     | |      | | |      | |     | | |   |  _|      | | |      | |     | | |  | |         | | |      | |     | | |   / ____ \   | | |    | |   _   | |\n\
| | _/ /    \ \_ | | |  _| |  \ \_  | | |    _| |_     | | |     _| |_    | | |  _| |_       | | |     _| |_    | | |  \ `.___.'\  | | |     _| |_    | | | _/ /    \ \_ | | |   _| |__/ |  | |\n\
| ||____|  |____|| | | |____| |___| | | |   |_____|    | | |    |_____|   | | | |_____|      | | |    |_____|   | | |   `._____.'  | | |    |_____|   | | ||____|  |____|| | |  |________|  | |\n\
| |              | | |              | | |              | | |              | | |              | | |              | | |              | | |              | | |              | | |              | |\n\
| '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' |\n\
 '----------------' '----------------' '----------------' '----------------' '----------------' '----------------' '----------------' '----------------' '----------------' '----------------'\n\
 .-----------------..----------------. .----------------. .----------------. .----------------. .----------------.   .-----------------..----------------. .----------------. .----------------. .----------------. .----------------. .----------------.\n\
| .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. | .--------------. |\n\
| | ____  _____  | | |  _________   | | | _____  _____ | | |  _______     | | |      __      | | |   _____      | | | | ____  _____  | | |  _________   | | |  _________   | | | _____  _____ | | |     ____     | | |  _______     | | |  ___  ____   | |\n\
| ||_   \|_   _| | | | |_   ___  |  | | ||_   _||_   _|| | | |_   __ \    | | |     /  \     | | |  |_   _|     | | | ||_   \|_   _| | | | |_   ___  |  | | | |  _   _  |  | | ||_   _||_   _|| | |   .'    `.   | | | |_   __ \    | | | |_  ||_  _|  | |\n\
| |  |   \ | |   | | |   | |_  \_|  | | |  | |    | |  | | |   | |__) |   | | |    / /\ \    | | |    | |       | | | |  |   \ | |   | | |   | |_  \_|  | | | |_/ | | \_|  | | |  | | /\ | |  | | |  /  .--.  \  | | |   | |__) |   | | |   | |_/ /    | |\n\
| |  | |\ \| |   | | |   |  _|  _   | | |  | '    ' |  | | |   |  __ /    | | |   / ____ \   | | |    | |   _   | | | |  | |\ \| |   | | |   |  _|  _   | | |     | |      | | |  | |/  \| |  | | |  | |    | |  | | |   |  __ /    | | |   |  __'.    | |\n\
| | _| |_\   |_  | | |  _| |___/ |  | | |   \ `--' /   | | |  _| |  \ \_  | | | _/ /    \ \_ | | |   _| |__/ |  | | | | _| |_\   |_  | | |  _| |___/ |  | | |    _| |_     | | |  |   /\   |  | | |  \  `--'  /  | | |  _| |  \ \_  | | |  _| |  \ \_  | |\n\
| ||_____|\____| | | | |_________|  | | |    `.__.'    | | | |____| |___| | | ||____|  |____|| | |  |________|  | | | ||_____|\____| | | | |_________|  | | |   |_____|    | | |  |__/  \__|  | | |   `.____.'   | | | |____| |___| | | | |____||____| | |\n\
| |              | | |              | | |              | | |              | | |              | | |              | | | |              | | |              | | |              | | |              | | |              | | |              | | |              | |\n\
| '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' | '--------------' |\n\
 '----------------' '----------------' '----------------' '----------------' '----------------' '----------------'   '----------------' '----------------' '----------------' '----------------' '----------------' '----------------' '----------------'\n"

colors = {
        'blue': '\033[94m',
        'pink': '\033[95m',
        'green': '\033[92m',
        }


def colorize(string, color):
    if not color in colors: return string
    return colors[color] + string + '\033[0m'
ann_image_build_cmds = [
                        "docker build",
                        "nvidia-docker build -t ann ."
                       ]
def build_cpu_image(menu_obj):
    output_bytes = client.build(path = "./", tag = "ann")
    for item in output_bytes:
        pretty_output = json.loads(item)
        pretty_output = pretty_output['stream']
        print(pretty_output, end='')
    display_menu_options(menu_obj)
    get_sub_menu_selection(menu_obj)
    #next_menu_selection(menu_obj)
    #input("Press [Enter] to continue")

def enter_cpu_container(menu_obj):
    print("In enter cpu container")
    return;

#have to set this to use nvidia-docker via the client runtime parameter (something like that)
def build_gpu_image():
    output_bytes = client.build(path = "./", tag = "ann")
    for item in output_bytes:
        pretty_output = json.loads(item)
        pretty_output = pretty_output['stream']
        print(pretty_output, end='')
    input("Press [Enter] to continue")

def enter_gpu_container():
    return;


def get_main_menu_selection(menu_obj):
    choice = input(">> ")
    # check for errors
    try:
        if int(choice) < 0 : 
            raise ValueError 
    except (ValueError, IndexError):
        pass
    return choice

def get_sub_menu_selection(menu_obj):
    choice = input(">> ")
    # check for errors
    try:
        if int(choice) < 0 : 
            raise ValueError 
        # Call the matching function
        list(menu_obj[int(choice)].values())[0](menu_obj)

    except (ValueError, IndexError):
        pass
    #return choice

def display_menu_options(menu_obj):
    display_item = ''
    for item in menu_obj:
        display_item += colorize("[" + str(menu_obj.index(item)) + "] ", 'blue') + list(item.keys())[0] + " "
    print ( display_item )


def next_menu_selection(menu_obj):
    display_menu_options(menu_obj)
    choice = get_main_menu_selection(menu_obj)
    if choice == "0" :
        print("In image menu selection")
        display_menu_options(ann_image_menu_choices)
        get_sub_menu_selection(ann_image_menu_choices)
    elif choice == "1":
        print("Here 2")
    elif choice == "2":
        print("Goodbye")
        exit()
    else:
        next_menu_selection(menu_obj)
'''
    #select either image or container options
    print(choice)
    if choice == 0:
        print ("choose image")
        next_menu_selection(ann_image_menu_choices)
        # choose image
    elif choice == 1:
        next_menu_selection(ann_container_menu_choices)
        print ("choose container")
        # choose container
    else:
        print("exiting")
        next_menu_selection(menu_obj)
'''

def main_menu():
    os.system('clear')
    print(colorize(header, 'blue'))
    next_menu_selection(ann_base_menu_choices)


def back_to_base_menu(message = None):
    main_menu()
def exit_menu_loop(message = None):
    exit()
    '''
    while True:
        os.system('clear')
        display_item = ''
        for item in ann_image_menu_choices:
            display_item += colorize("[" + str(ann_image_menu_choices.index(item)) + "] ", 'blue') + list(item.keys())[0] + " "
        print ( display_item )
        choice = input(">> ")
        try:
            if int(choice) < 0 : raise ValueError
            # Call the matching function
            list(ann_image_menu_choices[int(choice)].values())[0]()
        except (ValueError, IndexError):
            pass
       '''     
ann_image_menu_choices = [
                                {"Build Ann Image For Cpu": build_cpu_image},
                                {"Build Ann Image For Gpu": build_gpu_image},
                                {"Back To Main Menu": back_to_base_menu},
                                {"Exit": exit_menu_loop}
                         ]
ann_container_menu_choices = [
                                {"Enter Ann Cpu Container": enter_cpu_container},
                                {"Enter Ann Gpu Container": enter_gpu_container},
                                {"Back To Main Menu": back_to_base_menu},
                                {"Exit": exit_menu_loop}
                             ]
# need two functions 
ann_base_menu_choices = [
                            {"Display Image Options": display_menu_options},
                            {"Display Container Options": display_menu_options},
                            {"Exit": exit_menu_loop}
                        ]

if __name__ == "__main__":
    main_menu()

