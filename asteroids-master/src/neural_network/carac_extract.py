import numpy as np


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
        frame_data = []

        if ship is not None:
            ship_data = [ship.position.x, ship.position.y, ship.boundingRect.top, ship.boundingRect.bottom,
                         ship.boundingRect.left, ship.boundingRect.right]
            frame_data.append(ship_data)
            for rock in rock_list:
                rock_data = [rock.position.x, rock.position.y, rock.boundingRect.top, rock.boundingRect.bottom,
                             rock.boundingRect.left, rock.boundingRect.right]
                frame_data.append(rock_data)

            other_data = [number_life, score, 0, 0, 0, 0]
            frame_data.append(other_data)

            self.dataset.append(frame_data)
            self.ground_truth.append(input)

    def save_data(self):
        dataset_array = np.array(self.dataset)
        ground_truth = np.array(self.ground_truth)

        np.save('dataset.npy', dataset_array)  # save
        np.save('ground_truth.npy', ground_truth)  # save

    def load_data(self, dataset_path, groundtruth_path):
        self.dataset = np.load(dataset_path)  # load
        self.ground_truth = np.load(groundtruth_path)  # load

        return self.dataset, self.ground_truth
