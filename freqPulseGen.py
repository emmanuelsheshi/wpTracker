import math
import matplotlib.pyplot as plt



freqs = [i for i in range(1,3000)]
durations = []



def getFreqPair(allFreqz):

    for myfreq in allFreqz:
        duration = 360000/myfreq
        durations.append(duration)
    return durations

def getFreqPairSingle(allFreqz):
        duration = 360000/allFreqz
        return duration



def getFreqPairSingle2(allFreqz):
        duration = 1440000/allFreqz
        return duration

calculatedDurations = getFreqPair(freqs)

# print(calculatedDurations)
# print(freqs)
#
# plt.plot(freqs, calculatedDurations)
# plt.show()

print(getFreqPairSingle(50))