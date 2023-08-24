import random

if __name__ == "__main__":

    alls = []
    fo = open("./filelists/libri_train_clu_sid.txt", "r+")
    while True:
        try:
            message = fo.readline().strip()
        except Exception as e:
            print("nothing of except:", e)
            break
        if message == None:
            break
        if message == "":
            break
        alls.append(message)
    fo.close()

    valids = alls[:150]
    tests = alls[150:300]
    trains = alls[300:]

    random.shuffle(trains)

    fw = open("./filelists/singing_valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()

    fw = open("./filelists/singing_test.txt", "w", encoding="utf-8")
    for strs in tests:
        print(strs, file=fw)

    fw = open("./filelists/singing_train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)

    fw.close()
