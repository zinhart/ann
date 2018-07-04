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
def build_cpu_image():
    print("Building cpu image")
    output = client.build(path = "./", tag = "ann")
    for val in output:
        json_fmt = json.loads(val)
        pretty = json_fmt['stream']
        pretty = re.sub('\n','', pretty)
        pretty = re.sub('\r','', pretty)
        print(pretty)
    input("Press [Enter] to continue")

def build_gpu_image():
    print("Building gpu image")
    subprocess.call(ann_image_build_cmds[1])
    input("Press [Enter] to continue")

def main():
    while True:
        os.system('clear')
        # Print some badass ascii art header here !
       # print ( colorize(header, 'pink') ) 
       # print ( colorize('version 0.1\n', 'green') )
        for item in ann_image_menu_choices:
            print ( colorize("[" + str(ann_image_menu_choices.index(item)) + "] ", 'blue') + list(item.keys())[0] )
        choice = input(">> ")
        try:
            if int(choice) < 0 : raise ValueError
            # Call the matching function
            list(ann_image_menu_choices[int(choice)].values())[0]()
        except (ValueError, IndexError):
            pass
            
ann_image_menu_choices = [
                                {"Build Ann image for cpu-threaded": build_cpu_image},
                                {"Build Ann image for gpu-threaded": build_gpu_image},
                                {"Exit": exit}
                         ]

if __name__ == "__main__":
    print(colorize(header, 'blue'))
    main()

