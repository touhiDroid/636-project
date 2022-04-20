import glob

from math import sqrt

if __name__ == '__main__':
    allDataFiles = glob.glob('data/*')
    allDataFiles.sort()
    totalFiles = len(allDataFiles)
    with open('csi.csv', 'w') as csvOutFile:
        fc = 1
        for f in allDataFiles:
            escatVals = f.split(".")[0].split("_")
            if not f.endswith(".csv") or len(escatVals) < 5:
                continue
            # A valid CSV file is found. -> Let's parse the file-name for values of e, s, c, a & t
            # ---> NOPE, we're interested only in the activity, so ignoring environment, subject, class & trial values
            # escatVals[0][1] + ',' + escatVals[1][1:] + ',' + escatVals[2][1:] + ',' + escatVals[4][1:] + ','
            escatStr = str(int(escatVals[3][1:]))
            print('---------\n\nActivity: ', escatStr)
            with open(f, 'r') as csiFile:
                print(fc, '# Transforming file: ', f)
                lineCount = 1
                for csiLine in csiFile:
                    lineStr = escatStr
                    csiCols = csiLine.split(",")
                    if csiCols[0].startswith('time'):  # ignoring the title-row
                        continue
                    for i in range(13, len(csiCols)):  # reading only the csi (r-i pairs) values
                        try:
                            # X+Yi format: need to parse to amplitude=sqrt(X^2 + Y^2)
                            csiVals = csiCols[i][0:csiCols[i].rfind('i')].split("+")
                            x = float(csiVals[0])
                            y = float(csiVals[1])
                            lineStr += ",{:.2f}".format(sqrt(x * x + y * y))
                        except Exception as e:
                            print(e)
                    lineStr += '\n'
                    csvOutFile.write(lineStr)
                    lineCount += 1
                # Completed reading a single-CSV-file
                print(f'New CSI: {lineCount * (i - 13 + 1)} (Lines: {lineCount})\nProgress: {fc * 100 / totalFiles}%\n')
            fc += 1
        # Completed reading the entire data-folder
    print('Transformation Complete!')
