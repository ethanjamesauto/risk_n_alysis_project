import matplotlib.pyplot as plt

with open("400_length.txt", 'r') as fp:
    state = ""
    two_data = {}
    for line in fp:
        line = line.split("\n")[0]


        
        if ":" in line:
            state = line.split(":")[0]
            two_data[state] = []
        else:
            if line != '':
                two_data[state].append(float(line.replace("[","").replace("]","").replace(" ","")))

with open("400_l_5drop.txt", 'r') as fp:
    state = ""
    five_data = {}
    for line in fp:
        line = line.split("\n")[0]


        
        if ":" in line:
            state = line.split(":")[0]
            five_data[state] = []
        else:
            if line != '':
                five_data[state].append(float(line.replace("[","").replace("]","").replace(" ","")))

#print(data)

import numpy as np
# calculate rmse between rela and sigmoid data
re = np.array(two_data["real"])
si_2 = np.array(two_data["sigmoid"])
si_5 = np.array(five_data["sigmoid"])
rmse_2 = np.sqrt(np.mean((re - si_2)**2))
rmse_5 = np.sqrt(np.mean((re - si_5)**2))
print("rmse between real and sigmoid with dropout 0.2: ", rmse_2)
print("rmse between real and sigmoid with dropout 0.5: ", rmse_5)

plt.plot(five_data["real"][:200], color = "orange", label = "Real Stock Price")
plt.plot(five_data["sigmoid"][:200], color = "green", label = "dropout 0.5")
plt.plot(two_data["sigmoid"][:200], color = "red", label = "dropout 0.2")
#plt.plot(data["sigmoid"][:100], color = "red", label = "sigmoid")
plt.title("Stock Price Prediction using sigmoid activation with different dropouts")
plt.xlabel("Time")
plt.ylabel("Apple Stock Price")
plt.legend()
plt.show()
