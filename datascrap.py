from tkinter import *
import tkinter as tk
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import csv

nameteam = ""
list3 = []
testcode = """"""


def Webscrapper(url):
    global nameteam
    global L
    global b

    res = requests.get(url)
    res.encoding = "utf-8"

    soup = BeautifulSoup(res.text, "lxml")

    StoreNumHome = {}
    StoreNumAway = {}
    StoreNumDraw = {}
    StoreNumHomeAfter = {}
    StoreNumAwayAfter = {}
    StoreNumDrawAfter = {}
    StoreOdd1 = {}
    StoreHiLo1 = {}

    StoreOdd2 = {}
    StoreHiLo2 = {}
    for val in range(5):
        StoreNumHome["Case{0}".format(val + 1)] = 0
        StoreNumAway["Case{0}".format(val + 1)] = 0
        StoreNumDraw["Case{0}".format(val + 1)] = 0
        StoreNumHomeAfter["Case{0}".format(val + 1)] = 0
        StoreNumAwayAfter["Case{0}".format(val + 1)] = 0
        StoreNumDrawAfter["Case{0}".format(val + 1)] = 0
        StoreOdd1["Case{0}".format(val + 1)] = 0
        StoreHiLo1["Case{0}".format(val + 1)] = 0
        StoreOdd2["Case{0}".format(val + 1)] = 0
        StoreHiLo2["Case{0}".format(val + 1)] = 0

    for i in (8, 9, 10, 19, 20, 21, 30, 31, 32, 41, 42, 43, 52, 53, 54):
        soup2 = soup.find(id="main2").find_all("td")[i]
        string1 = str(soup2)
        string1 = string1.replace('<td width="8%">', "")
        string1 = string1.replace('<br/><span class="up">', " ")
        string1 = string1.replace('<br/><span class="down">', " ")
        string1 = string1.replace('<br/><span class="">', " ")
        string1 = string1.replace("</span></td>", "")
        string1 = string1.split()
        match i:
            case 8:
                StoreNumHome["Case1"] = string1[0]
                StoreNumHomeAfter["Case1"] = string1[1]
            case 9:
                StoreNumDraw["Case1"] = string1[0]
                StoreNumDrawAfter["Case1"] = string1[1]
            case 10:
                StoreNumAway["Case1"] = string1[0]
                StoreNumAwayAfter["Case1"] = string1[1]
            
            case 19:
                StoreNumHome["Case2"] = string1[0]
                StoreNumHomeAfter["Case2"] = string1[1]
            case 20:
                StoreNumDraw["Case2"] = string1[0]
                StoreNumDrawAfter["Case2"] = string1[1]
            case 21:
                StoreNumAway["Case2"] = string1[0]
                StoreNumAwayAfter["Case2"] = string1[1]

            case 30:
                StoreNumHome["Case3"] = string1[0]
                StoreNumHomeAfter["Case3"] = string1[1]
            case 31:
                StoreNumDraw["Case3"] = string1[0]
                StoreNumDrawAfter["Case3"] = string1[1]
            case 32:
                StoreNumAway["Case3"] = string1[0]
                StoreNumAwayAfter["Case3"] = string1[1]

            case 41:
                StoreNumHome["Case4"] = string1[0]
                StoreNumHomeAfter["Case4"] = string1[1]
            case 42:
                StoreNumDraw["Case4"] = string1[0]
                StoreNumDrawAfter["Case4"] = string1[1]
            case 43:
                StoreNumAway["Case4"] = string1[0]
                StoreNumAwayAfter["Case4"] = string1[1]

            case 52:
                StoreNumHome["Case5"] = string1[0]
                StoreNumHomeAfter["Case5"] = string1[1]
            case 53:
                StoreNumDraw["Case5"] = string1[0]
                StoreNumDrawAfter["Case5"] = string1[1]
            case 54:
                StoreNumAway["Case5"] = string1[0]
                StoreNumAwayAfter["Case5"] = string1[1]

    soup4 = soup.find(id="main2").find_all("tr")[0]
    string2 = str(soup4)
    string2 = string2.replace("", "")
    string2 = string2.replace('<font color="yellow">', "")
    string2 = string2.replace("</font>", "")
    string2 = string2.replace("</td>", "")
    string2 = string2.replace(" ", "")
    string2 = string2.replace('<td colspan="13" height="25">', "")
    string2 = string2.replace('<font color="yellow">', "")
    string2 = string2.replace("</font>", "")
    string2 = string2.replace("</td>", "")
    string2 = string2.replace("</tr>", "")
    string2 = string2.replace(" ", "")
    string2 = string2.split()
    
    gg = string2[0].replace(
        '<tralign="center"class="scoretitle"><tdcolspan="13"height="25">', ""
    )
    list1 = gg + " " + string2[1] + " " + string2[2]

    for i in (13, 16, 24, 27, 35, 38, 46, 49, 57, 60):
        #
        soup4 = soup.find(id="main2").find_all("td")[i]
        # ก้อนใหญ๋
        string3 = str(soup4)
        # ก้อนเล็ก
        soup5 = soup4.find("span")
        string4 = str(soup5)

        oddbefore = string3.replace("{}".format(soup5), "")
        oddbefore = oddbefore.replace('<td width="8%">', "")
        oddbefore = oddbefore.replace("<br/>", "")
        oddbefore = oddbefore.replace("</td>", "")
        find1 = oddbefore.find("/")
        if find1 < 0:
            numbefore = float(oddbefore)
        if find1 > 0:
            oddbefore = oddbefore.replace("/", " ")
            oddbefore = oddbefore.split()
            numbefore = (float(oddbefore[0]) + float(oddbefore[1])) / 2

        oddafter = string4.replace('<span class="up">', "")
        oddafter = oddafter.replace('<span class="">', "")
        oddafter = oddafter.replace('<span class="down">', "")
        oddafter = oddafter.replace("</span>", "")
        oddafter = oddafter.replace("</td>", "")
        find2 = oddafter.find("/")
        if find2 < 0:
            numafter = float(oddafter)
        if find2 > 0:
            oddafter = oddafter.replace("/", " ")
            oddafter = oddafter.split()
            numafter = (float(oddafter[0]) + float(oddafter[1])) / 2

        match i:
            case 13:
                StoreOdd1["Case1"] = float(numbefore)
                StoreOdd2["Case1"] = float(numafter)
            case 16:
                StoreHiLo1["Case1"] = float(numbefore)
                StoreHiLo2["Case1"] = float(numafter)

            case 24:
                StoreOdd1["Case2"] = float(numbefore)
                StoreOdd2["Case2"] = float(numafter)
            case 27:
                StoreHiLo1["Case2"] = float(numbefore)
                StoreHiLo2["Case2"] = float(numafter)

            case 35:
                StoreOdd1["Case3"] = float(numbefore)
                StoreOdd2["Case3"] = float(numafter)
            case 38:
                StoreHiLo1["Case3"] = float(numbefore)
                StoreHiLo2["Case3"] = float(numafter)

            case 46:
                StoreOdd1["Case4"] = float(numbefore)
                StoreOdd2["Case4"] = float(numafter)
            case 49:
                StoreHiLo1["Case4"] = float(numbefore)
                StoreHiLo2["Case4"] = float(numafter)

            case 57:
                StoreOdd1["Case5"] = float(numbefore)
                StoreOdd2["Case5"] = float(numafter)
            case 60:
                StoreHiLo1["Case5"] = float(numbefore)
                StoreHiLo2["Case5"] = float(numafter)
    # print((StoreOdd1["Case1"] + StoreOdd1["Case2"] +StoreOdd1["Case3"]  +StoreOdd1["Case4"] + StoreOdd1["Case5"])/5)
    # ค่าน้ำเฉลี่ยก่อนหน้า
    oddbeforeavg = (StoreOdd1["Case1"]+ StoreOdd1["Case2"]+ StoreOdd1["Case3"]+ StoreOdd1["Case4"]+ StoreOdd1["Case5"]) / 5
    print(oddbeforeavg)
    # ค่าน้ำเฉลี่ยตอนหลัง
    oddafteravg = (StoreOdd2["Case1"]+ StoreOdd2["Case2"]+ StoreOdd2["Case3"]+ StoreOdd2["Case4"]+ StoreOdd2["Case5"]) / 5
    # print(oddbeforeavg)
    hilobeforeavg = (StoreHiLo1["Case1"]+ StoreHiLo1["Case2"]+ StoreHiLo1["Case3"]+ StoreHiLo1["Case4"]+ StoreHiLo1["Case5"]) / 5
    # print(hilobeforeavg)
    hiloaftereavg = (StoreHiLo2["Case1"]+ StoreHiLo2["Case2"]+ StoreHiLo2["Case3"]+ StoreHiLo2["Case4"]+ StoreHiLo2["Case5"]) / 5
    # print(hiloaftereavg)
    homeavg = (float(StoreNumHome["Case1"])+ float(StoreNumHome["Case2"])+ float(StoreNumHome["Case3"])+ float(StoreNumHome["Case4"])+ float(StoreNumHome["Case5"])) / 5
    # print(homeavg)
    awayavg = (float(StoreNumAway["Case1"])
        + float(StoreNumAway["Case2"])
        + float(StoreNumAway["Case3"])
        + float(StoreNumAway["Case4"])
        + float(StoreNumAway["Case5"])
    ) / 5
    drawavg = (
        float(StoreNumDraw["Case1"])
        + float(StoreNumDraw["Case2"])
        + float(StoreNumDraw["Case3"])
        + float(StoreNumDraw["Case4"])
        + float(StoreNumDraw["Case5"])
    ) / 5
    homeavgAfter = (
        float(StoreNumHomeAfter["Case1"])
        + float(StoreNumHomeAfter["Case2"])
        + float(StoreNumHomeAfter["Case3"])
        + float(StoreNumHomeAfter["Case4"])
        + float(StoreNumHomeAfter["Case5"])
    ) / 5
    awayavgAfter = (
        float(StoreNumAwayAfter["Case1"])
        + float(StoreNumAwayAfter["Case2"])
        + float(StoreNumAwayAfter["Case3"])
        + float(StoreNumAwayAfter["Case4"])
        + float(StoreNumAwayAfter["Case5"])
    ) / 5
    drawavgAfter = (
        float(StoreNumDrawAfter["Case1"])
        + float(StoreNumDrawAfter["Case2"])
        + float(StoreNumDrawAfter["Case3"])
        + float(StoreNumDrawAfter["Case4"])
        + float(StoreNumDrawAfter["Case5"])
    ) / 5
    print(
        "%.3f" % homeavg,
        ",",
        "%.3f" % awayavg,
        ",",
        "%.3f" % drawavg,
        ",",
        "%.3f" % homeavgAfter,
        ",",
        "%.3f" % awayavgAfter,
        ",",
        "%.3f" % drawavgAfter,
        ",",
        "%.3f" % (homeavg - homeavgAfter),
        ",",
        "%.3f" % (awayavg - awayavgAfter),
        ",",
        "%.3f" % (drawavg - drawavgAfter),
        ",",
        oddbeforeavg,
        ",",
        oddafteravg,
        ",",
        "%.3f" % (oddbeforeavg - oddafteravg),
        ",",
        hilobeforeavg,
        ",",
        hiloaftereavg,
        ",",
        "%.3f" % (hilobeforeavg - hiloaftereavg),
        ",",
        "",
        list1,
        gg,
    )
    list2 = [
        [
            "%.3f" % homeavg,
            "%.3f" % awayavg,
            "%.3f" % drawavg,
            "%.3f" % homeavgAfter,
            "%.3f" % awayavgAfter,
            "%.3f" % drawavgAfter,
            "%.3f" % (homeavg - homeavgAfter),
            "%.3f" % (awayavg - awayavgAfter),
            "%.3f" % (drawavg - drawavgAfter),
            oddbeforeavg,
            oddafteravg,
            "%.3f" % (oddbeforeavg - oddafteravg),
            hilobeforeavg,
            hiloaftereavg,
            "%.3f" % (hilobeforeavg - hiloaftereavg),
            "",
            list1,
            gg,
        ]
    ]
    list3 = [
        "%.3f" % homeavg,
        "%.3f" % awayavg,
        "%.3f" % drawavg,
        "%.3f" % homeavgAfter,
        "%.3f" % awayavgAfter,
        "%.3f" % drawavgAfter,
        "%.3f" % (homeavg - homeavgAfter),
        "%.3f" % (awayavg - awayavgAfter),
        "%.3f" % (drawavg - drawavgAfter),
        oddbeforeavg,
        oddafteravg,
        "%.3f" % (oddbeforeavg - oddafteravg),
        hilobeforeavg,
        hiloaftereavg,
        "%.3f" % (hilobeforeavg - hiloaftereavg),
    ]
    b = np.array(list3).astype(float)
    nameteam = list1 + gg

    def write_to_csv(list_of_emails):
        with open("data2.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for email in list_of_emails:
                writer.writerow(email)
    write_to_csv(list2)   
    return 'Data has been saved to CSV file' , list2

def callPredict(url):
    global b
    global nameteam
    global x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
    result, list2 = Webscrapper(url)
    dataset = pd.read_csv(r"newmodel.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    dataset = pd.read_csv(r"newmodel.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1, random_state=0
    )

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    classifier.fit(X_train, y_train)

    from sklearn.neighbors import KNeighborsClassifier

    classifier1 = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    classifier1.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier2 = SVC(kernel="rbf", random_state=0)
    classifier2.fit(X_train, y_train)

    from sklearn.linear_model import LogisticRegression

    classifier3 = LogisticRegression(random_state=0)
    classifier3.fit(X_train, y_train)

    from sklearn.naive_bayes import GaussianNB

    classifier4 = GaussianNB()
    classifier4.fit(X_train, y_train)

    from sklearn.ensemble import RandomForestClassifier

    classifier5 = RandomForestClassifier(
        n_estimators=10, criterion="entropy", random_state=0
    )
    classifier5.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier6 = SVC(kernel="linear", random_state=0)
    classifier6.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier7 = SVC(kernel="poly", random_state=0)
    classifier7.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier8 = SVC(kernel="rbf", random_state=0)
    classifier8.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier9 = SVC(kernel="sigmoid", random_state=0)
    classifier9.fit(X_train, y_train)

    a = [b]

    y_pred = classifier.predict(a).item()
    y_pred2 = classifier1.predict(a).item()
    y_pred4 = classifier2.predict(a).item()
    y_pred6 = classifier3.predict(a).item()
    y_pred8 = classifier4.predict(a).item()
    y_pred10 = classifier5.predict(a).item()
    y_pred12 = classifier6.predict(a).item()
    y_pred14 = classifier7.predict(a).item()
    y_pred16 = classifier8.predict(a).item()
    y_pred18 = classifier9.predict(a).item()
    x1 = int(y_pred)
    x2 = int(y_pred2)
    x3 = int(y_pred4)
    x4 = int(y_pred6)
    x5 = int(y_pred8)
    x6 = int(y_pred10)
    x7 = int(y_pred12)
    x8 = int(y_pred14)
    x9 = int(y_pred16)
    x10 = int(y_pred18)
    testgroupby = [[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 3]]
    def write_to_csv(list_of_emails):
        with open("testgroupby.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for email in list_of_emails:
                writer.writerow(email)
    write_to_csv(testgroupby)   

# Webscrapper()
def callResult():
    # อ่านข้อมูลจากไฟล์ CSV
    global nameteam
    data = pd.read_csv("testgroupby.csv")
    global testcode
    global x1,x2,x3,x4,x5,x6,x7,x8,x9,x10
    # กำหนดลำดับที่ต้องการค้นหา
    search_sequence = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]

    # สร้างเงื่อนไขเพื่อค้นหาข้อมูลที่มีลำดับตามที่กำหนด
    condition = (data['group1'] == search_sequence[0]) & \
                (data['group2'] == search_sequence[1]) & \
                (data['group3'] == search_sequence[2]) & \
                (data['group4'] == search_sequence[3]) & \
                (data['group5'] == search_sequence[4]) & \
                (data['group6'] == search_sequence[5]) & \
                (data['group7'] == search_sequence[6]) & \
                (data['group8'] == search_sequence[7]) & \
                (data['group9'] == search_sequence[8]) & \
                (data['group10'] == search_sequence[9])

    # กรองข้อมูลด้วยเงื่อนไขที่กำหนด
    filtered_data = data[condition]

    # นับจำนวนข้อมูลที่มี result เป็น 0 หรือ 1
    result_counts = filtered_data['result'].value_counts()

    # แสดงผลลัพธ์
    print("ตำแหน่งที่ต้องการค้นหา:", search_sequence)
    print("จำนวนข้อมูลที่มีลำดับดังกล่าว:", len(filtered_data))
    print("จำนวนข้อมูลที่มี result เป็น 0:", result_counts.get(0, 0))
    print("จำนวนข้อมูลที่มี result เป็น 1:", result_counts.get(1, 0))
    testcode = nameteam + " have " + str(len(filtered_data)) + " case away " + str(result_counts.get(0, 0)) + " case home " + str(result_counts.get(1, 0)) + " case"
    return testcode

# เรียกใช้ฟังก์ชัน callResult เพื่อทดสอบ