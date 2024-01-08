# Data Mining: What is the effect of guided meditation on the distribution of sleep stages?

In this study it was researched whether guided meditation had an effect on the sleep stages of individuals, this was done by having the individuals listen to a singular guided meditation video directly before going to sleep. An example of which would be the video which could be found by following [this link](https://www.youtube.com/watch?v=ft-vhYwHzxw). This is just an example, as students were free to choose their own video of choice which as long as it was a guided meditation which was aimed at going to sleep. 

To be able to conduct the trial, there were two weeks of testing and two weeks of control collection. These should both be done consecutively, so either first two weeks of testing or first the two weeks of control sampling. However, due to measurement limitations, this was not always possible and was resolved by either using one week each or by splitting the control sampling to one week before and one week after the two weeks of the guided meditation.

As students there were of course some variables which might be seen as bottlenecks and which should be taken into account. For instance, there are varying variances and length of sleep schedules leading to the data having to be normalized with a metric like percentages. Next, alcohol usage causes a higher heart rate due to the fact that the liver needs to filter this. Thus, leading to possible distorted values in these nights. Moreover, there were a variety of different watches which were used to track the data. Where subjects 1-4 used the Xiaomi Smartband 7, subject 5 used the Apple Watch SE 2, which have different sensors and different ways of measuring. Therefore, it is possible that there is a deviation between the readings of these subjects. Finally, it became apparent that for some students there were some nights for which no sleeping data was recorded, which could either originate from a measurement error, or from the fact that a student forgot to wear his/her smartwatch

The data of the first subject can be found [here]()
The data of the second subject can be found [here]()
The data of the third subject can be found [here]()
The data of the fourth subject can be found [here]()
The data of the fifth subject can be found [here]()

The data is all produced in the same format. The data already contain a preprocessed pandas dataframe which can be opened using pd.read_pickle. The data in the dataframe had the following format:
* datetime (index): This is the date and the minute in which the bpm was recorded, for all other variables it reflects the sleep statistics of the date
* bpm: This is the number of beats per minute of the heart as recorded by the smartwatch
* deepSleepTime: This describes the number of minutes which the person spent in deep sleep during the night as recorded by the smartwatch
* shallowSleepTime: This describes the number of minutes which the person spent in shallow sleep during the night as recorded by the smartwatch
* wakeTime: This describes the number of minutes which the person spent in awake during the sleep as recorded by the smartwatch
* REMTime: This describes the number of minutes which the person spent in REM sleep during the night as recorded by the smartwatch
* start: This describes the exact date and time when the person started his/her sleep for the night, as recorded by the smartwatch
* stop: This describes the exact date and time when the person stopped his/her sleep for the night, as recorded by the smartwatch
* naps: This describes the number of minutes the person took naps during the day, as recorded by the smartwatch (this only include naps which are longer than 20 minutes)
* duration: The total duration of the sleep of this person as indicated in minutes
* label: The variable of interest, indicates whether the person has meditated before going to sleep
* day: This includes a number with the day of the month, so for instance, for October 31st, this would display 31
* month: This includes a number which represents the month, so for instance, for October 31st, this would display 10
* hour: This includes a number which represents the hour which the heart rate was monitored, so for instance, for October 31st 13:30, this would display 13
* week: This includes a number which represents the week, so for instance, for October 31st, this would display 44
* week: This includes a number which represents the day of the week, starting from 0 on Monday and ending at 6 on a Sunday, so for instance, for October 31st, which is a Tuesday, this would display 1. 
