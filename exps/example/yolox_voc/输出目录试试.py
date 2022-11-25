# import os
# a=os.getenv('Path',None)
# # print(a)
# data_dir = "..../default"
# # data_dir = "./data/VOCdevkit"
# print(data_dir)
class Debug:
    def mainProgram(self):
        x = int(input("Please input a integer: "))
        assert x > 5
        print(f"the value of x is: {x}")
        b=input('')

if __name__ == "__main__":
    main = Debug()
    main.mainProgram()