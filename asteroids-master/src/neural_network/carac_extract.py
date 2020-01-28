class Extract:
    def __init__(self):
        pass

    def display_rock_position(self,rock_list):
        for rock in rock_list:

            print('position',rock.position.x,rock.position.y)
            print('bouding box',rock.boundingRect)