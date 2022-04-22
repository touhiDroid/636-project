import json

if __name__ == '__main__':
    # Load the annotated time-ranges
    annotatedTimeRanges = []
    with open('../../data/annotations.csv', 'r') as annotatedCsvFile:
        for csvItem in annotatedCsvFile:
            splits = csvItem.split(",")
            startTime = float(splits[0])  # start-time-stamp
            endTime = float(splits[1])  # end-time-stamp
            # 2 -> class-name: "Mobile" or "Stationary" -> trimming the word to 1/0 to reduce the file-size
            itemClass = 1 if splits[4].strip() == "Mobile" else 0
            annotatedTimeRanges.append([startTime, endTime, itemClass])
    annotatedCsvFile.close()

    csiOutFile = open('../../data/csi.csv', 'w')
    dataCount = 0
    # 18,58,022 CSI data in '260-4.csi.json' file (2.28 GB)
    with open('../../data/128a-54.csi.json', 'r') as csiJsonFile:
        for csiJson in csiJsonFile:
            dataCount += 1
            joData = json.loads(csiJson)
            timeStamp = float(joData['t'])
            # Check whether timeStamp is within any allowed annotated data range or not
            outClass = -1
            for a in annotatedTimeRanges:
                if a[0] <= timeStamp <= a[1]:
                    outClass = int(a[2])
                    break
            if outClass == -1:
                continue
            print(str(dataCount) + ' ~ Rewriting Class: ' + ("Mobile" if outClass == 1 else "Stationary"))

            csiDataArray = joData['csi']  # 30x3 size 2D array of {"r": X, "i": Y} JSON objects
            # Write all 30x3x2 numbers in a
            for csiItem in csiDataArray:  # 30 x [ csiItem = 1D array of 3 {"r": X, "i": Y} JSON objects ]
                csiOutFile.write(str(csiItem[0]['r']) + ',' + str(csiItem[0]['i']) + ","
                                 + str(csiItem[1]['r']) + ',' + str(csiItem[1]['i']) + ","
                                 + str(csiItem[2]['r']) + ',' + str(csiItem[2]['i']) + ",")
            csiOutFile.write(str(outClass) + '\n')
    csiOutFile.close()
    csiJsonFile.close()