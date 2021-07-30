try:
    f = open(nn_filepath,'x')
except FileNotFoundError:
    f = open(nn_filepath,'w')
    f.close()
except FileExistsError:
    overwite = input("A network is already stored at this location, overwrite? y/n")
    if overwite != "y":
        f.close()
        return "quit to preserve previous file"
    else:
        f.close()
try:
    f = open(path,'x')
except FileNotFoundError:
    f = open(path,'w')
    f.close()
except FileExistsError:
    overwrite = input(str(path) + " already exists, overwrite? y/n")
    if overwrite != "y":
        f.close()
        return "quit to preserve previous file"
    else:
        f.close()
