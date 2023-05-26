"""
Creates an OSC bridge between Python and Pd.

- Receive the coordinates from user input in Pd
"""
import os

import pandas as pd
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME

# Connection parameters
HOST = "127.0.0.1"
IN_PORT = 1415
OUT_PORT = 1123


dataset_name = "babyslakh_20_1bar_4res"
embedding_name = "t-SNE_Drums"
embedding_path = os.path.join(
    DATASETS_DIR, dataset_name, PLOTS_DIRNAME, "pairspaces", f"{embedding_name}.csv"
)

# Dimensions of the gcanvas object in Pd
GCANVAS_SIZE = (300, 300)

if __name__ == "__main__":
    coord = [0, 0]
    quitFlag = [False]

    # Define a dispatcher to route incoming OSC messages to the right actions
    dispatcher = Dispatcher()

    def coord_message_handler(address, *args):
        """Handle messages starting with /coord"""
        x = args[0] / GCANVAS_SIZE[0]
        y = args[1] / GCANVAS_SIZE[1]
        print(x, y)

    def quit_message_handler(address, *args):
        """Handle messages starting with /quit"""
        quitFlag[0] = True
        print("Quitting!")

    def default_handler(address, *args):
        """Handle messages without a dedicated handler"""
        print(f"No action taken for message {address}: {args}")

    # Pass the handlers to the dispatcher
    dispatcher.map("/coord*", coord_message_handler)
    dispatcher.map("/quit*", quit_message_handler)
    dispatcher.set_default_handler(default_handler)

    # Start a UPD receiver and connect the dispatcher
    receiver = BlockingOSCUDPServer((HOST, IN_PORT), dispatcher)
    print(f"Listening on {HOST}:{IN_PORT}...")

    # Start the UPD sender client for sending messages to Pd
    osc_sender = SimpleUDPClient(HOST, OUT_PORT)

    # Send the coordinates of the points in the embedding spaces
    emb = pd.read_csv(embedding_path)
    osc_sender.send_message("/embedding/points", emb.values.tolist())

    while quitFlag[0] is False:
        receiver.handle_request()

        # duration = int(random.randrange(0, 1000))

        # Send Notes to pd (send pitch last to ensure syncing)
        # osc_sender.send_message("/coord", (vel, duration))
