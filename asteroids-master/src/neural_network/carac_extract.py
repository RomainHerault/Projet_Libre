import numpy as np
import pickle
from datetime import datetime


class Extract:
    def __init__(self):
        self.dataset = []
        self.ground_truth = []

    def display_rock_position(self, rock_list):
        print('number of rocks', len(rock_list))
        for rock in rock_list:
            print('position', rock.position.x, rock.position.y)
            print('bouding box', rock.boundingRect)

    def display_ship(self, ship):
        if ship is not None:
            print('ship position ', ship.position.x, ship.position.y)
            print('ship bouding box ', ship.boundingRect)

    def display_number_life(self, number_life):
        print('number of life ', number_life)

    def display_score(self, score):
        print('score ', score)

    def get_data(self, ship, rock_list, number_life, score, input):
        asteroids_number = 48
        frame_data = np.zeros((asteroids_number + 2, 6))

        if ship is not None:
            ship_data = [ship.position.x, ship.position.y,
                         ship.boundingRect.top, ship.boundingRect.bottom,
                         ship.boundingRect.left, ship.boundingRect.right]
            frame_data[0] = ship_data
            for i in range(len(rock_list)):
                if i < asteroids_number:
                    rock = rock_list[i]
                    rock_data = [rock.position.x, rock.position.y,
                                 rock.boundingRect.top, rock.boundingRect.bottom,
                                 rock.boundingRect.left, rock.boundingRect.right]
                    frame_data[i+1] = rock_data
                else :
                    print("Y'A TROP D'ASTEROIIIIIDS !!! AU SECOUUUURS !!!!")
            # for rock in rock_list:
            #     rock_data = [rock.position.x, rock.position.y,
            #                  rock.boundingRect.top, rock.boundingRect.bottom,
            #                  rock.boundingRect.left, rock.boundingRect.right]
            #     frame_data.append(rock_data)

            other_data = [number_life, score, 0, 0, 0, 0]
            frame_data[-1] = other_data

            self.dataset.append(frame_data)
            self.ground_truth.append(np.array(input))

    def save_data(self):
        dataset_array = np.array(self.dataset)
        ground_truth = np.array(self.ground_truth)
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)

        filename_string = "SavedData/dataset_" + now.strftime(
            "%d-%m-%Y_%H-%M-%S")

        final = [dataset_array, ground_truth]

        pickle_out = open(filename_string, "wb")
        pickle.dump(final, pickle_out)
        pickle_out.close()

        # np.save('dataset.npy', dataset_array)  # save
        # np.save('ground_truth.npy', ground_truth)  # save

    def load_data(self, dataset_path):
        infile = open(dataset_path, 'rb')
        final = pickle.load(infile)
        infile.close()

        return final[0], final[1]
