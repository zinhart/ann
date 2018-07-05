import os, sys, subprocess, docker, dockerpty, re, json
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

def clear_screen(menu_obj = None):
    os.system('clear')
    if menu_obj != None:
        display_menu_options(menu_obj)
        get_sub_menu_selection(menu_obj)


def build_cpu_image(menu_obj):
    output_bytes = client.build(path = "./", tag = "ann")
    for item in output_bytes:
        pretty_output = json.loads(item)
        pretty_output = pretty_output['stream']
        print(pretty_output, end='')
    display_menu_options(menu_obj)
    get_sub_menu_selection(menu_obj)
    #input("Press [Enter] to continue")

def enter_cpu_container(menu_obj):
    print("In enter cpu container")
    #dockerpty
    return;

#have to set this to use nvidia-docker via the client runtime parameter (something like that)
def build_gpu_image(menu_obj):
    output_bytes = client.build(path = "./", tag = "ann")
    for item in output_bytes:
        pretty_output = json.loads(item)
        pretty_output = pretty_output['stream']
        print(pretty_output, end='')
    display_menu_options(menu_obj)
    get_sub_menu_selection(menu_obj)
    #input("Press [Enter] to continue")

def enter_gpu_container():
    #dockerpty
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

def display_menu_options(menu_obj):
    display_item = ''
    for item in menu_obj:
        display_item += colorize("[" + str(menu_obj.index(item)) + "] ", 'blue') + list(item.keys())[0] + " "
    print ( display_item )

'''
Recursively loops through menu options
'''
def next_menu_selection(menu_obj):
    display_menu_options(menu_obj)
    choice = get_main_menu_selection(menu_obj)
    
    if choice == str(len(menu_obj)-1):# always check for exit condition first
        exit_menu_loop()
    elif choice == "0" :
        print("In image menu selection")
        display_menu_options(ann_image_menu_choices)
        get_sub_menu_selection(ann_image_menu_choices)
    elif choice == "1":
        display_menu_options(ann_container_menu_choices)
        get_sub_menu_selection(ann_container_menu_choices)
        print("Here 2")
    else:
        next_menu_selection(menu_obj)

def main_menu():
    clear_screen()
    print(colorize(header, 'blue'))
    next_menu_selection(ann_base_menu_choices)

def back_to_base_menu(message = None):
    main_menu()

def exit_menu_loop(message = None):
    exit()

def list_images(menu_obj):
    print (client.images(all=True))
    display_menu_options(menu_obj)
    get_sub_menu_selection(menu_obj)

def list_containers(menu_obj):
    print (client.containers(all=True))
    display_menu_options(menu_obj)
    get_sub_menu_selection(menu_obj)



ann_image_menu_choices = [
                                {"List Images": list_images},
                                {"Clear": clear_screen},
                                {"Build Ann Image For Cpu": build_cpu_image},
                                {"Build Ann Image For Gpu": build_gpu_image},
                                {"Back To Main Menu": back_to_base_menu},
                                {"Exit": exit_menu_loop}
                         ]
ann_container_menu_choices = [
                                {"List Containers": list_containers},
                                {"Clear": clear_screen},
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

