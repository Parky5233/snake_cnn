import requests
import os
import csv

# future improvement, automated query and download to folder?

if __name__ == '__main__':
    os.chdir('snake_images')
    snake_images = []
    count = 0
    species_csvs = [fName for fName in os.listdir() if fName.endswith(".csv")]
    data_size = 1500
    train_size = (int)(data_size * 0.85)
    test_size = data_size-train_size
    for species in species_csvs:  # for number of species in folder
        print(os.getcwd())
        with open(species) as snake_data:

            training = True
            test_done = False
            os.chdir("train")
            cur_snake = species.split(".")[0]
            if not os.path.isdir(cur_snake):
                os.mkdir(cur_snake)
            os.chdir(cur_snake)
            if(training == True and len([name for name in os.listdir('.') if os.path.isfile(name)])>=train_size):
                training = False
                count = train_size
                os.chdir("../../test")
                if not os.path.isdir(cur_snake):
                    os.mkdir(cur_snake)
                os.chdir(cur_snake)
            if(training == False and len([name for name in os.listdir('.') if os.path.isfile(name)])>=test_size):
                print("all files already downloaded for "+cur_snake)
                os.chdir('../..')
                test_done = True
            if(training == False and test_done == True):
                continue
            else:
                data_read = csv.reader(snake_data, delimiter=",")
                count = 1
                next(data_read)
                for line in data_read:
                    #print(line)
                    link = line[1]
                    img_fName = cur_snake+'_'+line[0]+".jpg"
                    r = requests.get(link, stream=True)
                    if r.status_code == 200:
                        with open(img_fName, 'wb') as save:
                            for chunk in r:
                                save.write(chunk)
                    else:
                        print("this image failed to save\n" + link)
                        count -= 1
                    if training:
                        if count == train_size:
                            training = False
                            os.chdir("../../test")
                            if not os.path.isdir(cur_snake):
                                os.mkdir(cur_snake)
                            os.chdir(cur_snake)
                            print("downloading testing")
                        count += 1
                    elif count == data_size:
                        os.chdir('..')
                        break
                    else:
                        count += 1

                    if (count) % 100 == 0:
                        #print(os.getcwd())
                        print((str)(count) + " images of "+(str)(data_size)+" downloaded")
        os.chdir('..')
        print("Done downloading: " + species)

print("Done downloading photos")
