import numpy as np
class Extract:
    def __init__(self):
        self.dataset = []

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

    def get_data(self, ship, rock_list, number_life, score):
        frame_data = []
        if ship is not None :
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
            array = np.array(self.dataset)

            #print(array.shape)
