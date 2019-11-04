import sys
import os
import statistics
import re

# Alright lemme start the cleaning of this incomplete messes of a db

# This is the variable that holds everything.
incompletelines = 0
incompleteindexes = list()
fulllines = 0
datalines = list()
letterlines = list()
hardCodeDict = {
    0: ("low", "mon", "monday","jan","january","small","unacc",0),
    1: ("med", "medium", "tue", "tuesday","feb","february","acc",1),
    2: ("high", "wed", "wednesday","mar","march","big","good",2),
    3: ("vhigh","thu", "thursday","apr","april","vgood",3),
    4: ("fri", "friday","may",4),
    5: ("sat", "saturday","jun","june",5),
    6: ("sun", "sunday","jul","july","5more","more",6),
    7: ("aug", "august",7),
    8: ("sep", "september",8),
    9: ("oct", "october", 9),
    10: ("nov", "november", 10),
    11: ("dec", "december",11)
}

class DataLine(object):
    content = list()
    continuity = True
    noncontidx = list()


    def __init__(self, loadedcontent):
        self.content = loadedcontent
        global letterlines
        #Second is the loop to detect missing content.
        for x in range(0, len(self.content)):
            if self.content[x] is None:
                self.content[x] = "?"
            if self.content[x] == "?":
                # If the content has something that is null or ?, note where it is and indicate something is missing.
                self.continuity = False
                self.noncontidx.append(x)
            if not(self.content[x].strip().isdigit() or isfloat(self.content[x])): #Look if the content is not numbers
                try:
                   letterlines.index(x)
                except (ValueError):
                    letterlines.append(x)
        self.content[-1] = self.content[-1].strip()

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def isint(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

def ComaPreproc(line):
    return line.replace(';',',')

def daryonmode_outputtext():
    print("Verbose Mode has been activated. Everything the program does is gonna be output on the console.")
    for y in range(0, len(datalines) - 1):
        stringer = "Line " + str(y) + " :"
        for z in range(0, len(datalines[y].content) - 2):
            stringer += str(datalines[y].content[z]) + ", "
        stringer += str(datalines[y].content[-1])
        print(stringer + "\n")

#Looks into the Hard Coding Dictionary for what value to return
def hardCodedPreProc(thingToLookFor):
    global hardCodeDict
    for key in hardCodeDict.keys():
        if thingToLookFor in hardCodeDict[key]:
            if any("verbose" in s for s in sys.argv):
                print("Harcoding " + thingToLookFor + " as " + str(key))
            return key
    return -1

def main():
    global incompletelines
    global fulllines
    global datalines
    global incompleteindexes
    global letterlines
    filepath = ""
    isCSV = False
    line1_csv = ""
    print("The Janitor. Data Preprocessor by Othmane Filali Benaceur. (use -help for more info.)")
    if any("-help" in s for s in sys.argv):
        print("""Use: input the filename after the Python Script.
-verbose - Activates Verbose mode\n-forceX - Subtitute X for a method (median/delete) to force using it\n-manualamp - Manually replace all missing data.
-linenum - Add a line numbering at the start of each line of Data\n-columnswitch:X - Switches the column X to last position.(For Classes)""")
        exit(0)
    if len(sys.argv) < 2:
        print("Error, no valid argument.")
        exit(-1)
        #Open the file.
    with open(sys.argv[1], "r+") as file:
        if file is None:
            print("Error, non-existant file or empty file.\n")
            exit(-1)
        filepath = os.path.dirname(sys.argv[1])
        if sys.argv[1].split('.')[-1] == "csv":
            isCSV = True
        lines = file.readlines()
        #Create a Dataline for each line and append it to the datalines.
    for line in lines:
        if isCSV:
            line1_csv = line
            isCSV = False
            continue
        if(any(';' in s for s in line)):
            line = ComaPreproc(line)
        classifiedline = DataLine(line.split(','))
        if not classifiedline.continuity:
            #have a list will all the locations for the incomplete arrays.
            incompletelines += 1
            incompleteindexes.append(len(datalines))
        fulllines += 1
        datalines.append(classifiedline)
        #Verbose Mode.

    #Before trying to do anything, we will need to get rid of any kind of non-numbers.
    for index in letterlines: #if you get inside this, that means one of the columns has numbers
        #print("Index: " + str(index))
        letterinput = []
        for line in datalines:
            if not(isint(line.content[index]) or isfloat(line.content[index])): #First see if said content is digit and skip it otherwise
                try:
                    letterinput.index(line.content[index])
                except (ValueError):
                    letterinput.append(line.content[index])
                # at this state, it should be appended, so you can use it on the spot
                hardCodeKey = hardCodedPreProc(line.content[index])
                #print("Line: " + str(hardCodeKey))
                if hardCodeKey != -1:
                    line.content[index] = hardCodeKey
                else:
                    line.content[index] = letterinput.index(line.content[index])
        #letterlines.remove(index)

    if any("-verbose" in s for s in sys.argv):
        daryonmode_outputtext()
    print("Incomplete Lines: " + str(incompletelines))
    print("Complete Lines: " + str(fulllines))
    incompletePercentage = round(incompletelines/fulllines*100,2)
    print("Incomplete percentage: " + str(incompletePercentage) + "%")
    #The user can still force the use of a specific technique by writing it in the command line.
    if any("-manualamp" in s for s in sys.argv):
        amp = input("Please enter the value for the Manual Amputation > ")
        for i in range(0,len(datalines)):
            for j in range(0,len(datalines[i].content)):
                if datalines[i].content[j] == '?':
                    datalines[i].content[j] = amp
    elif (any("-forcemedian" in s for s in sys.argv) or incompletePercentage > 5) and not any("-forcedelete" in s for s in sys.argv):
        print("Median Technique selected.")
        #Let's create an array for each column so we can amputate with Medians.
        mediancalcarrs = []
        for i in range(0,len(datalines[0].content)):
            templist = []
            for j in range(0,len(datalines)):
                if j not in incompleteindexes:
                    templist.append(datalines[j].content[i])
                mediancalcarrs.append(templist)
        mediancalced = [] #This is the medians that need to be inserted.
        for list in mediancalcarrs:
            mediancalced.append(statistics.median(list))
        if any("-verbose" in s for s in sys.argv):
            print("Incomplete Indexes: " + str(incompleteindexes))
        for idx in range(0,len(incompleteindexes)):
            for idx2 in range(len(datalines[incompleteindexes[idx]].noncontidx)):
                datalines[incompleteindexes[idx]].content[datalines[incompleteindexes[idx]].noncontidx[idx2]] = str(mediancalced[datalines[incompleteindexes[idx]].noncontidx[idx2]])
            datalines[incompleteindexes[idx]].continuity = True
            if any("-verbose" in s for s in sys.argv):
                print("Dataline "+ str(incompleteindexes[idx]) + " :" + str(datalines[incompleteindexes[idx]].content))
    elif incompletePercentage > 0:
        print("Deletion Technique selected.")
        print("Dataline Length: " + str(len(datalines)))
        for line in datalines:
            if not line.continuity:
                datalines.remove(line)
                fulllines -= 1
    else:
        print("No Imputation to do.")
    print("Dataline Final Length: " + str(len(datalines)))
    #Print the content of the Datalines back into an output file.
    file = open(filepath + "\\output_" + sys.argv[1].split('\\')[-1],"w+")
    for y in range(0, len(datalines) - 1):
        stringer = ""
        if any("-linenum" in s for s in sys.argv):
            stringer = str((y+1)) + ", "
        if any("-columnswitch:" in s for s in sys.argv):
            try:
                num = 0
                for i in range(len(sys.argv)):
                    if "-columnswitch:" in sys.argv:
                        num = i
                        break
                columnToSwitch = int(sys.argv[i].split(':')[-1])
            except ValueError:
                columnToSwitch = -1
            for z in range(0, len(datalines[y].content) - 1):
                if not (z == columnToSwitch):
                    stringer += str(datalines[y].content[z]) + ", "
            stringer += str(datalines[y].content[-1]) + ", "
            stringer += str(datalines[y].content[columnToSwitch])
            stringer += "\n"
            file.write(stringer)
        else:
            for z in range(0, len(datalines[y].content) - 1):
                stringer += str(datalines[y].content[z]) + ", "
            stringer += str(datalines[y].content[-1])
            stringer += "\n"
            file.write(stringer)
    #file.write(line1_csv)
    file.close()


if __name__ == "__main__":
    main()


