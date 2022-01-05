---
jupyter:
  colab:
    name: Project_sol_7\_Ramesh_Ronak_Srihari_Animesh.ipynb
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.5
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code colab="{\"height\":258,\"base_uri\":\"https://localhost:8080/\"}" id="usPnk4UqlNEg" outputId="b04a7c48-2496-4c4e-9ea2-5f76a620fddf"}
``` {.python}
#######################################
###!@0 START INIT ENVIRONMENT
from google.colab import drive
drive.mount('/content/drive')
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
#!wget -q https://mirrors.estointernet.in/apache/spark/spark-3.0.3/spark-3.0.3-bin-hadoop2.7.tgz -P /content/drive/MyDrive # link wrong in blog
!tar xf /content/drive/Shareddrives/DA231-2021-Aug-Public/spark-3.0.3-bin-hadoop2.7.tgz
!pip install -q findspark
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.3-bin-hadoop2.7"

###!@0 END INIT ENVIRONMENT

#######################################
###!@1 START OF PYSPARK INIT
# Provides findspark.init() to make pyspark importable as a regular library.
# Resource : https://pypi.org/project/findspark/
import findspark
findspark.init()
findspark.find()
from pyspark.sql import SparkSession
spark = SparkSession.builder\
         .master("local")\
         .appName("Colab")\
         .config('spark.ui.port', '4050')\
         .getOrCreate()
spark
# Spark is ready to go within Colab!
###!@1 END OF PYSPARK INIT
```

::: {.output .stream .stdout}
    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
:::

::: {.output .execute_result execution_count="51"}
```{=html}
            <div>
                <p><b>SparkSession - in-memory</b></p>
                
        <div>
            <p><b>SparkContext</b></p>

            <p><a href="http://4c714b86bbb2:4050">Spark UI</a></p>

            <dl>
              <dt>Version</dt>
                <dd><code>v3.0.3</code></dd>
              <dt>Master</dt>
                <dd><code>local</code></dd>
              <dt>AppName</dt>
                <dd><code>Colab</code></dd>
            </dl>
        </div>
        
            </div>
        
```
:::

::: {.output .stream .stdout}
    time: 12.3 s (started: 2021-12-10 17:27:48 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="24NrVumOkDtJ" outputId="4e609a32-2b2d-4eca-d7a8-96cd43b56554"}
``` {.python}
#######################################
###!@2 START OF DEFINING INPUT FILES
#pfile = "/content/drive/MyDrive/US_youtube_trending_data.csv"
pfile = "/content/drive/MyDrive/youtube_trending_video_dataset/US_youtube_trending_data.csv"
jfile = "/content/drive/MyDrive/youtube_trending_video_dataset/US_category_id.json"

#jfile = "/content/drive/MyDrive/US_category_id.json"
###!@2 END OF DEFINING INPUT FILES

#######################################
###!@3 START OF LOADING DATA
from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import date
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

!pip install ipython-autotime
%load_ext autotime
##########################


# load USA YT data from CSV
print("USA VIDEO DATA SET")
usadf = spark.read.option("header",True).option("inferSchema",True).option("multiline","true").csv(pfile).where(col("video_id").isNotNull())
usadf.show(5)

# Load US Json
print("USA VIDEO DATA SET CATEGORY and ID")
ytvusjdf = spark.read.option("multiline","true").json(jfile).select(explode("items").alias("itemsExplode")).select("itemsExplode.*").withColumn("Category", col("snippet").title).drop("etag", "kind", "snippet")
ytvusjdf.show(10)


#cfile = "/content/drive/MyDrive/YouTubeProjectSem1/US_youtube_trending_data.csv"
#jfile = "/content/drive/MyDrive/YouTubeProjectSem1/US_category_id.json"

#c1file = "/content/drive/MyDrive/YouTubeProjectSem1/GB_youtube_trending_data.csv"
#j1file = "/content/drive/MyDrive/YouTubeProjectSem1/GB_category_id.json"

#c2file = "/content/drive/MyDrive/YouTubeProjectSem1/IN_youtube_trending_data.csv"
#j2file = "/content/drive/MyDrive/YouTubeProjectSem1/IN_category_id.json"

cfile = "/content/drive/MyDrive/youtube_trending_video_dataset/US_youtube_trending_data.csv"
jfile = "/content/drive/MyDrive/youtube_trending_video_dataset/US_category_id.json"

c1file = "/content/drive/MyDrive/youtube_trending_video_dataset/GB_youtube_trending_data.csv"
j1file = "/content/drive/MyDrive/youtube_trending_video_dataset/GB_category_id.json"

c2file = "/content/drive/MyDrive/youtube_trending_video_dataset/IN_youtube_trending_data.csv"
j2file = "/content/drive/MyDrive/youtube_trending_video_dataset/IN_category_id.json"
#######################################
###!@3 START OF LOADING DATA
##########################

# load US data from CSV
uscdf = usadf
#uscdf = spark.read.option("header",True).option("inferSchema",True).option("multiline","true").csv(cfile).where(col("video_id").isNotNull())
gbcdf = spark.read.option("header",True).option("inferSchema",True).option("multiline","true").csv(c1file).where(col("video_id").isNotNull())
incdf = spark.read.option("header",True).option("inferSchema",True).option("multiline","true").csv(c2file).where(col("video_id").isNotNull())
print("USA VIDEO DATA SET")
uscdf.show(5)
print("Great Britain VIDEO DATA SET")
gbcdf.show(5)
print("India VIDEO DATA SET")
incdf.show(5)

# # Load US Json
usjdf = ytvusjdf
#usjdf = spark.read.option("multiline","true").json(jfile).select(explode("items").alias("itemsExplode")).select("itemsExplode.*").withColumn("Category", col("snippet").title).drop("etag", "kind", "snippet")
gbjdf = spark.read.option("multiline","true").json(j1file).select(explode("items").alias("itemsExplode")).select("itemsExplode.*").withColumn("Category", col("snippet").title).drop("etag", "kind", "snippet")
injdf = spark.read.option("multiline","true").json(j2file).select(explode("items").alias("itemsExplode")).select("itemsExplode.*").withColumn("Category", col("snippet").title).drop("etag", "kind", "snippet")
print("USA VIDEO DATA SET CATEGORY and ID")
usjdf.show(5)
print("Great Britain VIDEO DATA SET CATEGORY and ID")
gbjdf.show(5)
print("India VIDEO DATA SET CATEGORY and ID")
injdf.show(5)
```

::: {.output .stream .stdout}
    Requirement already satisfied: ipython-autotime in /usr/local/lib/python3.7/dist-packages (0.3.1)
    Requirement already satisfied: ipython in /usr/local/lib/python3.7/dist-packages (from ipython-autotime) (5.5.0)
    Requirement already satisfied: decorator in /usr/local/lib/python3.7/dist-packages (from ipython->ipython-autotime) (4.4.2)
    Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.7/dist-packages (from ipython->ipython-autotime) (5.1.1)
    Requirement already satisfied: pexpect in /usr/local/lib/python3.7/dist-packages (from ipython->ipython-autotime) (4.8.0)
    Requirement already satisfied: pygments in /usr/local/lib/python3.7/dist-packages (from ipython->ipython-autotime) (2.6.1)
    Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.7/dist-packages (from ipython->ipython-autotime) (0.8.1)
    Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.7/dist-packages (from ipython->ipython-autotime) (1.0.18)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.7/dist-packages (from ipython->ipython-autotime) (0.7.5)
    Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.7/dist-packages (from ipython->ipython-autotime) (57.4.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython->ipython-autotime) (0.2.5)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython->ipython-autotime) (1.15.0)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.7/dist-packages (from pexpect->ipython->ipython-autotime) (0.7.0)
    The autotime extension is already loaded. To reload it, use:
      %reload_ext autotime
    USA VIDEO DATA SET
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    |   video_id|               title|        publishedAt|           channelId| channelTitle|categoryId|      trending_date|                tags|view_count| likes|dislikes|comment_count|      thumbnail_link|comments_disabled|ratings_disabled|         description|
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    |3C66w5Z0ixs|I ASKED HER TO BE...|2020-08-11 19:20:14|UCvtRTOMP2TqYqu51...|     Brawadis|        22|2020-08-12 00:00:00|brawadis|prank|ba...|   1514614|156908|    5855|        35313|https://i.ytimg.c...|            false|           false|SUBSCRIBE to BRAW...|
    |M9Pmf9AB4Mo|Apex Legends | St...|2020-08-11 17:00:10|UC0ZV6M2THA81QT9h...| Apex Legends|        20|2020-08-12 00:00:00|Apex Legends|Apex...|   2381688|146739|    2794|        16549|https://i.ytimg.c...|            false|           false|While running her...|
    |J78aPJ3VyNs|I left youtube fo...|2020-08-11 16:34:06|UCYzPXprvl5Y-Sf0g...|jacksepticeye|        24|2020-08-12 00:00:00|jacksepticeye|fun...|   2038853|353787|    2628|        40221|https://i.ytimg.c...|            false|           false|I left youtube fo...|
    |kXLn3HkpjaA|XXL 2020 Freshman...|2020-08-11 16:38:55|UCbg_UMjlHJg_19SZ...|          XXL|        10|2020-08-12 00:00:00|xxl freshman|xxl ...|    496771| 23251|    1856|         7647|https://i.ytimg.c...|            false|           false|Subscribe to XXL ...|
    |VIUo6yapDbc|Ultimate DIY Home...|2020-08-11 15:10:05|UCDVPcEbVLQgLZX0R...|     Mr. Kate|        26|2020-08-12 00:00:00|The LaBrant Famil...|   1123889| 45802|     964|         2196|https://i.ytimg.c...|            false|           false|Transforming The ...|
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    only showing top 5 rows

    USA VIDEO DATA SET CATEGORY and ID
    +---+----------------+
    | id|        Category|
    +---+----------------+
    |  1|Film & Animation|
    |  2|Autos & Vehicles|
    | 10|           Music|
    | 15|  Pets & Animals|
    | 17|          Sports|
    | 18|    Short Movies|
    | 19| Travel & Events|
    | 20|          Gaming|
    | 21|   Videoblogging|
    | 22|  People & Blogs|
    +---+----------------+
    only showing top 10 rows

    USA VIDEO DATA SET
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    |   video_id|               title|        publishedAt|           channelId| channelTitle|categoryId|      trending_date|                tags|view_count| likes|dislikes|comment_count|      thumbnail_link|comments_disabled|ratings_disabled|         description|
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    |3C66w5Z0ixs|I ASKED HER TO BE...|2020-08-11 19:20:14|UCvtRTOMP2TqYqu51...|     Brawadis|        22|2020-08-12 00:00:00|brawadis|prank|ba...|   1514614|156908|    5855|        35313|https://i.ytimg.c...|            false|           false|SUBSCRIBE to BRAW...|
    |M9Pmf9AB4Mo|Apex Legends | St...|2020-08-11 17:00:10|UC0ZV6M2THA81QT9h...| Apex Legends|        20|2020-08-12 00:00:00|Apex Legends|Apex...|   2381688|146739|    2794|        16549|https://i.ytimg.c...|            false|           false|While running her...|
    |J78aPJ3VyNs|I left youtube fo...|2020-08-11 16:34:06|UCYzPXprvl5Y-Sf0g...|jacksepticeye|        24|2020-08-12 00:00:00|jacksepticeye|fun...|   2038853|353787|    2628|        40221|https://i.ytimg.c...|            false|           false|I left youtube fo...|
    |kXLn3HkpjaA|XXL 2020 Freshman...|2020-08-11 16:38:55|UCbg_UMjlHJg_19SZ...|          XXL|        10|2020-08-12 00:00:00|xxl freshman|xxl ...|    496771| 23251|    1856|         7647|https://i.ytimg.c...|            false|           false|Subscribe to XXL ...|
    |VIUo6yapDbc|Ultimate DIY Home...|2020-08-11 15:10:05|UCDVPcEbVLQgLZX0R...|     Mr. Kate|        26|2020-08-12 00:00:00|The LaBrant Famil...|   1123889| 45802|     964|         2196|https://i.ytimg.c...|            false|           false|Transforming The ...|
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    only showing top 5 rows

    Great Britain VIDEO DATA SET
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    |   video_id|               title|        publishedAt|           channelId| channelTitle|categoryId|      trending_date|                tags|view_count| likes|dislikes|comment_count|      thumbnail_link|comments_disabled|ratings_disabled|         description|
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    |J78aPJ3VyNs|I left youtube fo...|2020-08-11 16:34:06|UCYzPXprvl5Y-Sf0g...|jacksepticeye|        24|2020-08-12 00:00:00|jacksepticeye|fun...|   2038853|353790|    2628|        40228|https://i.ytimg.c...|            false|           false|I left youtube fo...|
    |9nidKH8cM38|TAXI CAB SLAYER K...|2020-08-11 20:00:45|UCFMbX7frWZfuWdjA...|Eleanor Neale|        27|2020-08-12 00:00:00|eleanor|neale|ele...|    236830| 16423|     209|         1642|https://i.ytimg.c...|            false|           false|The first 1000 pe...|
    |M9Pmf9AB4Mo|Apex Legends | St...|2020-08-11 17:00:10|UC0ZV6M2THA81QT9h...| Apex Legends|        20|2020-08-12 00:00:00|Apex Legends|Apex...|   2381688|146739|    2794|        16549|https://i.ytimg.c...|            false|           false|While running her...|
    |kgUV1MaD_M8|Nines - Clout (Of...|2020-08-10 18:30:28|UCvDkzrj8ZPlBqRd6...|        Nines|        24|2020-08-12 00:00:00|Nines|Trapper of ...|    613785| 37567|     669|         2101|https://i.ytimg.c...|            false|           false|Nines - Clout (Of...|
    |49Z6Mv4_WCA|i don't know what...|2020-08-11 20:24:34|UCtinbF-Q-fVthA0q...| CaseyNeistat|        22|2020-08-12 00:00:00|              [None]|    940036| 87113|    1860|         7052|https://i.ytimg.c...|            false|           false|ssend love to my ...|
    +-----------+--------------------+-------------------+--------------------+-------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    only showing top 5 rows

    India VIDEO DATA SET
    +-----------+--------------------+-------------------+--------------------+--------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    |   video_id|               title|        publishedAt|           channelId|  channelTitle|categoryId|      trending_date|                tags|view_count| likes|dislikes|comment_count|      thumbnail_link|comments_disabled|ratings_disabled|         description|
    +-----------+--------------------+-------------------+--------------------+--------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    |Iot0eF6EoNA|Sadak 2 | Officia...|2020-08-12 04:31:41|UCGqvJPRcv7aVFun-...|  FoxStarHindi|        24|2020-08-12 00:00:00|sadak|sadak 2|mah...|   9885899|224925| 3979409|       350210|https://i.ytimg.c...|            false|           false|Three Streams. Th...|
    |x-KbnJ9fvJc|Kya Baat Aa : Kar...|2020-08-11 09:00:11|UCm9SZAl03Rev9sFw...|Rehaan Records|        10|2020-08-12 00:00:00|              [None]|  11308046|655450|   33242|       405146|https://i.ytimg.c...|            false|           false|Singer/Lyrics: Ka...|
    |KX06ksuS6Xo|Diljit Dosanjh: C...|2020-08-11 07:30:02|UCZRdNleCgW-BGUJf...|Diljit Dosanjh|        10|2020-08-12 00:00:00|clash diljit dosa...|   9140911|296533|    6179|        30058|https://i.ytimg.c...|            false|           false|CLASH official mu...|
    |UsMRgnTcchY|Dil Ko Maine Di K...|2020-08-10 05:30:49|UCq-Fj5jknLsUf-MW...|      T-Series|        10|2020-08-12 00:00:00|hindi songs|2020 ...|  23564512|743931|   84162|       136942|https://i.ytimg.c...|            false|           false|Gulshan Kumar and...|
    |WNSEXJJhKTU|Baarish (Official...|2020-08-11 05:30:13|UCye6Oz0mg46S362L...| VYRLOriginals|        10|2020-08-12 00:00:00|VYRL Original|Moh...|   6783649|268817|    8798|        22984|https://i.ytimg.c...|            false|           false|VYRL Originals br...|
    +-----------+--------------------+-------------------+--------------------+--------------+----------+-------------------+--------------------+----------+------+--------+-------------+--------------------+-----------------+----------------+--------------------+
    only showing top 5 rows

    USA VIDEO DATA SET CATEGORY and ID
    +---+----------------+
    | id|        Category|
    +---+----------------+
    |  1|Film & Animation|
    |  2|Autos & Vehicles|
    | 10|           Music|
    | 15|  Pets & Animals|
    | 17|          Sports|
    +---+----------------+
    only showing top 5 rows

    Great Britain VIDEO DATA SET CATEGORY and ID
    +---+----------------+
    | id|        Category|
    +---+----------------+
    |  1|Film & Animation|
    |  2|Autos & Vehicles|
    | 10|           Music|
    | 15|  Pets & Animals|
    | 17|          Sports|
    +---+----------------+
    only showing top 5 rows

    India VIDEO DATA SET CATEGORY and ID
    +---+----------------+
    | id|        Category|
    +---+----------------+
    |  1|Film & Animation|
    |  2|Autos & Vehicles|
    | 10|           Music|
    | 15|  Pets & Animals|
    | 17|          Sports|
    +---+----------------+
    only showing top 5 rows

    time: 9.81 s (started: 2021-12-10 17:28:01 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="YzWdkrOoph8j" outputId="d0d72a0f-9709-4413-85c0-85748737a96a"}
``` {.python}
# 1. EDA of trending video metrics:
# a) : How many number of views/comments/likes videos had when they first became trending? Visualization through histograms and plots.
# Note: If ratings_disabled is True than no Likes and Dislikes values

cleandf_1a = usadf.withColumn("view_count", col("view_count").cast("double")).withColumn("likes", col("likes").cast("double")).\
            withColumn("trending_date", col("trending_date").cast("date")).withColumn("comment_count", col("comment_count").cast("double"))\
            .withColumn("dislikes", col("dislikes").cast("double"))\
            .drop("thumbnail_link", "description", "tags") #,"comments_disabled","ratings_disabled")

firsttrendingdf_1a = cleandf_1a.select("video_id", expr("date(trending_date)")).\
             groupBy("video_id").agg(F.min("trending_date")).withColumnRenamed("min(trending_date)","tdate")\
             .withColumnRenamed("video_id","vid")

finaldf_1a = cleandf_1a.join(firsttrendingdf_1a, ((cleandf_1a.trending_date == firsttrendingdf_1a.tdate) & \
                            (cleandf_1a.video_id == firsttrendingdf_1a.vid))).drop("vid", "tdate")
finaldf_1a.show(10)
finaldf_1a.printSchema()
```

::: {.output .stream .stdout}
    +-----------+--------------------+-------------------+--------------------+--------------------+----------+-------------+----------+--------+--------+-------------+-----------------+----------------+
    |   video_id|               title|        publishedAt|           channelId|        channelTitle|categoryId|trending_date|view_count|   likes|dislikes|comment_count|comments_disabled|ratings_disabled|
    +-----------+--------------------+-------------------+--------------------+--------------------+----------+-------------+----------+--------+--------+-------------+-----------------+----------------+
    |5WjcDji3xYc|Honest Trailers |...|2020-08-11 17:03:59|UCOpcACMWblDls9Z6...|      Screen Junkies|         1|   2020-08-12|  833369.0| 50181.0|  1120.0|       4634.0|            false|           false|
    |84lMEGPUmi4|Is it time for Me...|2020-08-14 22:07:44|UC4i_9WvfPRTuRWEa...|            BT Sport|        17|   2020-08-15| 1150891.0| 27498.0|   455.0|       4261.0|            false|           false|
    |fqT81qdPpOw|P2 - Speedway (Of...|2020-08-14 16:00:11|UCn8wUiFIZsxbnvsi...|         P2istheName|        24|   2020-08-15|  899123.0|130706.0|  2222.0|      27659.0|            false|           false|
    |sa5rGfFWhN8|OMG! 2021 Ram 150...|2020-08-17 16:00:10|UCO-85LYfB61OP4SR...| The Fast Lane Truck|         2|   2020-08-18|  246430.0|  5697.0|   227.0|       1705.0|            false|           false|
    |5UmW4uDEIcs|Dyeing a wig for ...|2020-08-25 20:47:53|UCoziFm3M4sHDq1kk...|           Glam&Gore|        24|   2020-08-26|  469105.0| 77896.0|   198.0|       3307.0|            false|           false|
    |5UUdn_4Esck|PlayStation Plus ...|2020-08-26 15:30:00|UC6yzV_xgKn8r77Fk...|  PlayStation Access|        20|   2020-08-28|  387424.0|  8282.0|  2052.0|       1767.0|            false|           false|
    |3JxF7DuJzjc|ROCKETS at THUNDE...|2020-09-01 03:58:10|UCWJ2lWNubArHWmf3...|                 NBA|        17|   2020-09-01| 1174561.0| 10178.0|   478.0|       2768.0|            false|           false|
    |MORpybHP4po|Black Clover - Op...|2020-09-01 10:35:22|UCf2L21tpe1P-Y4Qj...|      Crunchyroll FR|        24|   2020-09-02|  362108.0| 27136.0|   442.0|       4102.0|            false|           false|
    |zq0iCh-Eink|Jelly Roll - Hous...|2020-09-10 17:00:10|UCtyzzW6rIQiGP7gf...|          Jelly Roll|        10|   2020-09-11|  204211.0| 20493.0|   169.0|       1694.0|            false|           false|
    |--SvHNpSvpk|YoungBoy Never Br...|2020-09-11 15:00:00|UClW4jraMKz6Qj69l...|YoungBoy Never Br...|        10|   2020-09-12| 1736660.0|127830.0|  2242.0|      11037.0|            false|           false|
    +-----------+--------------------+-------------------+--------------------+--------------------+----------+-------------+----------+--------+--------+-------------+-----------------+----------------+
    only showing top 10 rows

    root
     |-- video_id: string (nullable = true)
     |-- title: string (nullable = true)
     |-- publishedAt: timestamp (nullable = true)
     |-- channelId: string (nullable = true)
     |-- channelTitle: string (nullable = true)
     |-- categoryId: integer (nullable = true)
     |-- trending_date: date (nullable = true)
     |-- view_count: double (nullable = true)
     |-- likes: double (nullable = true)
     |-- dislikes: double (nullable = true)
     |-- comment_count: double (nullable = true)
     |-- comments_disabled: boolean (nullable = true)
     |-- ratings_disabled: boolean (nullable = true)

    time: 7.45 s (started: 2021-12-10 17:28:10 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="kKU0EYRTeKzb" outputId="8480081e-1c20-4b5c-f4f2-1b8274d1f32e"}
``` {.python}
## Avg. number of Likes for trending for first time 97666 (~1 Lakh) 
avglikesfortrendingdf_1a = finaldf_1a.agg({"likes": "avg"})
print("Avg. number of Likes for trending for first time = ", avglikesfortrendingdf_1a.collect()[0][0])
avglikesfortrendingdf_1a.show()

## Avg. number of Views for trending for first time 1344952 (~1.34 Million) 
avgviewsfortrendingdf_1a = finaldf_1a.agg({"view_count": "avg"})
print("Avg. number of Views for trending for first time = ", avgviewsfortrendingdf_1a.collect()[0][0])
avgviewsfortrendingdf_1a.show()

## Avg. number of Comments for trending for first time 9581 (~10 K) 
avgcommentfortrendingdf_1a = finaldf_1a.agg({"comment_count": "avg"})
print("Avg. number of Comments for trending for first time = ", avgcommentfortrendingdf_1a.collect()[0][0])
avgcommentfortrendingdf_1a.show()
```

::: {.output .stream .stdout}
    Avg. number of Likes for trending for first time =  97771.64861310378
    +-----------------+
    |       avg(likes)|
    +-----------------+
    |97771.64861310378|
    +-----------------+

    Avg. number of Views for trending for first time =  1358940.5845289335
    +------------------+
    |   avg(view_count)|
    +------------------+
    |1358940.5845289335|
    +------------------+

    Avg. number of Comments for trending for first time =  9470.932329029172
    +------------------+
    |avg(comment_count)|
    +------------------+
    | 9470.932329029172|
    +------------------+

    time: 49.4 s (started: 2021-12-10 17:28:18 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="5Q2i-9hZ3F7J" outputId="05a68b26-005d-4236-e458-bfaac2199102"}
``` {.python}
# MAX LIKES when first became trending
## "BTS (방탄소년단) 'Butter' Official MV" hosted by "HYBE LABELS" has highest number of Likes (7110071) and Comments(3.4M) when they first became trending. [Date: 2021-05-21  Views: 67.1M]
maxlikesdf_1a = finaldf_1a.orderBy("likes", ascending=False)
print("====== "+str(maxlikesdf_1a.collect()[0][1]) + " has HIGHEST number of LIKES : " + str(maxlikesdf_1a.collect()[0][8])+" ======")
maxlikesdf_1a.show(1, truncate=False)

# MAX Comments when first became trending
maxcommentsdf_1a = finaldf_1a.orderBy("comment_count", ascending=False)
print("====== "+ str(maxcommentsdf_1a.collect()[0][1]) + " has got HIGHEST number of COMMENTS : " + str(maxcommentsdf_1a.collect()[0][10])+" ======") 
maxcommentsdf_1a.show(1, truncate=False)

# MAX Views when first became trending
## "LISA - 'LALISA' M/V " by BLACKPINK has highets number of views (85.9 Million) when they first became trending. [Date: 2021-09-11  Likes: 5921316, Comments: 1.95M]
maxviewsdf_1a = finaldf_1a.orderBy("view_count", ascending=False)
print("====== "+ str(maxviewsdf_1a.collect()[0][1]) + " has HIGHEST number of VIEWS : " + str(maxviewsdf_1a.collect()[0][7])+ " ======")
maxviewsdf_1a.show(1, truncate=False)

# MAX Dislikes when first became trending
maxdislikesdf_1a = finaldf_1a.orderBy("dislikes", ascending=False)
## "BLACKPINK - 'Ice Cream (with Selena Gomez)' M/V" has highest dislikes (405329) when first became trending.
print("====== "+ str(maxdislikesdf_1a.collect()[0][1]) + " has MAXIMUM Dislikes : " + str(maxdislikesdf_1a.collect()[0][9])+ " ======")
maxdislikesdf_1a.show(1, truncate=False)
```

::: {.output .stream .stdout}
    ====== BTS (방탄소년단) 'Butter' Official MV has HIGHEST number of LIKES : 7110071.0 ======
    +-----------+-------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    |video_id   |title                                |publishedAt        |channelId               |channelTitle|categoryId|trending_date|view_count |likes    |dislikes|comment_count|comments_disabled|ratings_disabled|
    +-----------+-------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    |WMweEpGlu_U|BTS (방탄소년단) 'Butter' Official MV|2021-05-21 03:46:13|UC3IZKseVpdzPSBaWxBxundA|HYBE LABELS |10        |2021-05-21   |6.7111752E7|7110071.0|8998.0  |3400291.0    |false            |false           |
    +-----------+-------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    only showing top 1 row

    ====== BTS (방탄소년단) 'Butter' Official MV has got HIGHEST number of COMMENTS : 3400291.0 ======
    +-----------+-------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    |video_id   |title                                |publishedAt        |channelId               |channelTitle|categoryId|trending_date|view_count |likes    |dislikes|comment_count|comments_disabled|ratings_disabled|
    +-----------+-------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    |WMweEpGlu_U|BTS (방탄소년단) 'Butter' Official MV|2021-05-21 03:46:13|UC3IZKseVpdzPSBaWxBxundA|HYBE LABELS |10        |2021-05-21   |6.7111752E7|7110071.0|8998.0  |3400291.0    |false            |false           |
    +-----------+-------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    only showing top 1 row

    ====== LISA - 'LALISA' M/V has HIGHEST number of VIEWS : 85890366.0 ======
    +-----------+-------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    |video_id   |title              |publishedAt        |channelId               |channelTitle|categoryId|trending_date|view_count |likes    |dislikes|comment_count|comments_disabled|ratings_disabled|
    +-----------+-------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    |awkkyBH2zEo|LISA - 'LALISA' M/V|2021-09-10 04:00:13|UCOmHUn--16B90oW2L6FRR3A|BLACKPINK   |10        |2021-09-11   |8.5890366E7|5921316.0|38624.0 |1958529.0    |false            |false           |
    +-----------+-------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    only showing top 1 row

    ====== BLACKPINK - 'Ice Cream (with Selena Gomez)' M/V has MAXIMUM Dislikes : 405329.0 ======
    +-----------+-----------------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    |video_id   |title                                          |publishedAt        |channelId               |channelTitle|categoryId|trending_date|view_count |likes    |dislikes|comment_count|comments_disabled|ratings_disabled|
    +-----------+-----------------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    |vRXZj0DzXIA|BLACKPINK - 'Ice Cream (with Selena Gomez)' M/V|2020-08-28 04:00:11|UCOmHUn--16B90oW2L6FRR3A|BLACKPINK   |10        |2020-08-28   |5.1234434E7|5912778.0|405329.0|1847794.0    |false            |false           |
    +-----------+-----------------------------------------------+-------------------+------------------------+------------+----------+-------------+-----------+---------+--------+-------------+-----------------+----------------+
    only showing top 1 row

    time: 1min 36s (started: 2021-12-10 17:29:07 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="11HoAnjC3QNY" outputId="f86b1979-182a-4c09-f707-603689d94c36"}
``` {.python}
## MINIMUM :
## "Leading the Charge | Circle K" has only 17 likes , 6 dislikes and 10 comments when it became first trending.

mindlikesdf_1a = finaldf_1a.where(col("ratings_disabled") == "false").orderBy("likes", ascending=True)
print("====== "+str(mindlikesdf_1a.collect()[0][1]) + " has MINIMUM number of LIKES : " + str(mindlikesdf_1a.collect()[0][8])+" ======")
mindlikesdf_1a.show(1, truncate=False)

mindislikesdf_1a = finaldf_1a.where(col("ratings_disabled") == "false").orderBy("dislikes", ascending=True)
print("====== "+str(mindislikesdf_1a.collect()[0][1]) + " has MINIMUM number of DISLIKES : " + str(mindislikesdf_1a.collect()[0][9])+" ======")
mindislikesdf_1a.show(1, truncate=False)

mincommentsdf_1a = finaldf_1a.where(col("comments_disabled") == "false").select("video_id","title","channelTitle","comment_count","comments_disabled").orderBy("comment_count", ascending=True)
## 43 Videos have 0 comments when they first became trending.
mincommcount = mincommentsdf_1a.where(col("comment_count") == "0.0").count()
#print(mincommcount)
print("====== "+str(mincommcount) + " Videos have 0 comments when they first became trending. ======")
mincommentsdf_1a.show(4, truncate=False)

minviewsdf_1a = finaldf_1a.select("video_id","title","channelTitle","view_count").orderBy("view_count", ascending=True)
## 11 Video have 0 views when they first became trending.
minviewscount = minviewsdf_1a.where(col("view_count") == "0.0").count()
print("====== "+str(minviewscount) + " Videos have 0 views when they first became trending. ======")
minviewsdf_1a.show(4, truncate=False)
```

::: {.output .stream .stdout}
    ====== Leading the Charge | Circle K has MINIMUM number of LIKES : 17.0 ======
    +-----------+-----------------------------+-------------------+------------------------+------------+----------+-------------+----------+-----+--------+-------------+-----------------+----------------+
    |video_id   |title                        |publishedAt        |channelId               |channelTitle|categoryId|trending_date|view_count|likes|dislikes|comment_count|comments_disabled|ratings_disabled|
    +-----------+-----------------------------+-------------------+------------------------+------------+----------+-------------+----------+-----+--------+-------------+-----------------+----------------+
    |cfP1eeNPT_E|Leading the Charge | Circle K|2021-02-07 06:37:52|UC7kPuM7FmDnHjJHvJIPJP_Q|Circle K    |2         |2021-02-09   |1531323.0 |17.0 |6.0     |10.0         |false            |false           |
    +-----------+-----------------------------+-------------------+------------------------+------------+----------+-------------+----------+-----+--------+-------------+-----------------+----------------+
    only showing top 1 row

    ====== Leading the Charge | Circle K has MINIMUM number of DISLIKES : 6.0 ======
    +-----------+-----------------------------+-------------------+------------------------+------------+----------+-------------+----------+-----+--------+-------------+-----------------+----------------+
    |video_id   |title                        |publishedAt        |channelId               |channelTitle|categoryId|trending_date|view_count|likes|dislikes|comment_count|comments_disabled|ratings_disabled|
    +-----------+-----------------------------+-------------------+------------------------+------------+----------+-------------+----------+-----+--------+-------------+-----------------+----------------+
    |cfP1eeNPT_E|Leading the Charge | Circle K|2021-02-07 06:37:52|UC7kPuM7FmDnHjJHvJIPJP_Q|Circle K    |2         |2021-02-09   |1531323.0 |17.0 |6.0     |10.0         |false            |false           |
    +-----------+-----------------------------+-------------------+------------------------+------------+----------+-------------+----------+-----+--------+-------------+-----------------+----------------+
    only showing top 1 row

    ====== 43 Videos have 0 comments when they first became trending. ======
    +-----------+---------------------------------------------------------+------------------------------------+-------------+-----------------+
    |video_id   |title                                                    |channelTitle                        |comment_count|comments_disabled|
    +-----------+---------------------------------------------------------+------------------------------------+-------------+-----------------+
    |ehQal7cSIpE|Live 2020 Election Results                               |PBS NewsHour                        |0.0          |false            |
    |rbtybA7xT2M|Pokémon: Twilight Wings—The Gathering of Stars           |The Official Pokémon YouTube channel|0.0          |false            |
    |7AP86CYeR30|HOTEL TRANSYLVANIA: TRANSFORMANIA - Official Trailer (HD)|Sony Pictures Entertainment         |0.0          |false            |
    |--smQkLRmrY|Machine Gun Kelly - Downfalls High (Teaser Trailer)      |Machine Gun Kelly                   |0.0          |false            |
    +-----------+---------------------------------------------------------+------------------------------------+-------------+-----------------+
    only showing top 4 rows

    ====== 11 Videos have 0 views when they first became trending. ======
    +-----------+------------------------------------------------------------+-----------------+----------+
    |video_id   |title                                                       |channelTitle     |view_count|
    +-----------+------------------------------------------------------------+-----------------+----------+
    |BxOEj8ZeX2g|Tim Bergling's 32nd Birthday                                |GoogleDoodles    |0.0       |
    |mCY4b6GGkb4|The Funeral of The Duke of Edinburgh                        |The Royal Family |0.0       |
    |r7nYQXsxJdU|HBCU Homecoming 2020: Meet Me On The Yard                   |YouTube Originals|0.0       |
    |HcSwBJY7Xew|Watch The Weeknd and create short videos on the YouTube app.|YouTube          |0.0       |
    +-----------+------------------------------------------------------------+-----------------+----------+
    only showing top 4 rows

    time: 1min 20s (started: 2021-12-10 17:30:44 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="M4DNKV_8-XL6" outputId="41bf8294-3c7f-4210-f14a-3da0900ef67a"}
``` {.python}
VDFHIST1_1A = maxlikesdf_1a.select("likes").orderBy("likes", ascending=False)
VDFHIST1_1AOUT = [data[0] for data in VDFHIST1_1A.select('likes').collect()]
```

::: {.output .stream .stdout}
    time: 13.7 s (started: 2021-12-10 17:32:04 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":812,\"base_uri\":\"https://localhost:8080/\"}" id="Iuy9S5OwzfR5" outputId="85c8dfa9-5ae2-4767-9195-6686e163cdbb"}
``` {.python}
# Histogram for X: Number of likes (Vs) Y: number of Videos per bin/bucket.

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, ax = plt.subplots(figsize=(15,10))
ax.hist(VDFHIST1_1AOUT)
ax.set_title("Likes (Vs) Number of videos")
ax.set_xlabel('Likes')
ax.set_ylabel('Number of videos')
plt.yscale('log')
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/86de15cc47f6a482e4b58f06e9ddd705f7d12a7c.png)
:::

::: {.output .stream .stdout}
    time: 723 ms (started: 2021-12-10 17:32:18 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":812,\"base_uri\":\"https://localhost:8080/\"}" id="cuaOIVsvJmmh" outputId="a15fdcd4-7baa-4186-b03b-2dc6949c0369"}
``` {.python}
# Histogram for X: Number of Views (Vs) Y: number of videos per bin/bucket.
VDFHIST1VIEW_1A = maxviewsdf_1a.select("view_count").orderBy("view_count", ascending=False)
VDFHIST1VIEW_1AOUT = [data[0] for data in VDFHIST1VIEW_1A.select('view_count').collect()]

fig, ax = plt.subplots(figsize=(15,10))
ax.hist(VDFHIST1VIEW_1AOUT)
ax.set_title("Views (Vs) Number of videos")
ax.set_xlabel('Views')
ax.set_ylabel('Number of videos')
plt.yscale('log')
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/c77a98a9625c591490dd0c762423a8fb7a85a89b.png)
:::

::: {.output .stream .stdout}
    time: 14 s (started: 2021-12-10 17:32:19 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":812,\"base_uri\":\"https://localhost:8080/\"}" id="w5Cn0Cf7KMtA" outputId="dc161ede-0046-4510-b4f4-a0ecc01d722f"}
``` {.python}
# Histogram for X: Number of Comments (Vs) Y: number of videos per bin/bucket.
VDFHIST1COMM_1A = maxcommentsdf_1a.select("comment_count").orderBy("comment_count", ascending=False)
VDFHIST1COMM_1AOUT = [data[0] for data in VDFHIST1COMM_1A.select('comment_count').collect()]

fig, ax = plt.subplots(figsize=(15,10))
ax.hist(VDFHIST1COMM_1AOUT)
ax.set_title("Comments (Vs) Number of videos")
ax.set_xlabel('Comments')
ax.set_ylabel('Number of videos')
plt.yscale('log')
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/e9410661fac2a2aad7abd2fb81bccd9bb34280e8.png)
:::

::: {.output .stream .stdout}
    time: 13.5 s (started: 2021-12-10 17:32:33 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="9IEUbt3FmBiR" outputId="d37b8b2f-d707-4559-886b-60f5e48ee508"}
``` {.python}
# b) How long does it take a Video to become trending for the first time? Videos from which category has a longer trend? 
inputdf_1b = finaldf_1a.withColumn("publishedDate", expr("date(publishedAt)")).withColumn("timetakentotrend", expr("trending_date - publishedDate"))\
             .where(col("timetakentotrend").isNotNull()).withColumn("timetakentotrend",col("timetakentotrend").cast(StringType())).drop("channelId","comments_disabled","ratings_disabled")

split_timetaken_1b = F.split(inputdf_1b['timetakentotrend'], ' ')

daystakentotrenddf_1b = inputdf_1b.withColumn("daystakentotrend", split_timetaken_1b.getItem(0))\
            .withColumn("daystakentotrend",col("daystakentotrend").cast(IntegerType())).drop("timetakentotrend","")

maxdaystotrenddf_1b = daystakentotrenddf_1b.orderBy("daystakentotrend", ascending=False)
## Maximum it took 27 days for a video to trend from its publish date. (3 videos : All from Walmart)
print("====== Maximum it took "+ str(maxdaystotrenddf_1b.collect()[0][11]) + " days for a video to trend from its publish date. ======")
maxdaystotrenddf_1b.show(4, truncate=False)

## 1087 videos Started trending on same day as Published.
zerodaystotrenddf_1b = daystakentotrenddf_1b.where(col("daystakentotrend") == 0)
print("====== "+ str(zerodaystotrenddf_1b.count()) + " videos Started trending on same day as Published. ======")
zerodaystotrenddf_1b.show(5)
#zerodaystotrenddf_1b.count()

## Trending videos from Category ID 22 ("People & Blogs") took longer time to trend compared to other categories.
longertrendbycatdf_1b = daystakentotrenddf_1b.groupBy("categoryId").agg(F.avg("daystakentotrend"))\
                        .withColumnRenamed("avg(daystakentotrend)", "avg_daystakentotrend").orderBy("avg_daystakentotrend", ascending=False)

# Add Category Name from JASON File
categorynamedf_1b = ytvusjdf.join(longertrendbycatdf_1b, (longertrendbycatdf_1b.categoryId == ytvusjdf.id))\
                    .drop("id").orderBy("avg_daystakentotrend", ascending=False)
print("====== Trending videos from Category : "+ str(categorynamedf_1b.collect()[0][0]) + " took longer time to trend compared to other categories. ======")
categorynamedf_1b.show(20, truncate=False)
```

::: {.output .stream .stdout}
    ====== Maximum it took 27 days for a video to trend from its publish date. ======
    +-----------+-----------------------------------------------------------+-------------------+------------+----------+-------------+-----------+-------+--------+-------------+-------------+----------------+
    |video_id   |title                                                      |publishedAt        |channelTitle|categoryId|trending_date|view_count |likes  |dislikes|comment_count|publishedDate|daystakentotrend|
    +-----------+-----------------------------------------------------------+-------------------+------------+----------+-------------+-----------+-------+--------+-------------+-------------+----------------+
    |TR0I0STLyws|A different kind of membership. Walmart+ | Mobile scan & go|2020-09-15 13:00:11|Walmart     |22        |2020-10-12   |117988.0   |410.0  |283.0   |0.0          |2020-09-15   |27              |
    |izVdUXx5zaU|Walmart Back to College | However you study                |2020-08-06 17:10:23|Walmart     |22        |2020-09-02   |1205139.0  |24.0   |19.0    |16.0         |2020-08-06   |27              |
    |MeYkO5vkakg|A different kind of membership. Walmart+ | The Cox Family  |2020-09-15 13:00:05|Walmart     |22        |2020-10-12   |1.1624348E7|9273.0 |5636.0  |0.0          |2020-09-15   |27              |
    |Zsfo9ECSEtA|how to open packaged scissors with no scissors             |2020-10-13 16:00:09|Alex Martens|23        |2020-11-08   |467864.0   |14478.0|968.0   |629.0        |2020-10-13   |26              |
    +-----------+-----------------------------------------------------------+-------------------+------------+----------+-------------+-----------+-------+--------+-------------+-------------+----------------+
    only showing top 4 rows

    ====== 1092 videos Started trending on same day as Published. ======
    +-----------+--------------------+-------------------+--------------+----------+-------------+----------+--------+--------+-------------+-------------+----------------+
    |   video_id|               title|        publishedAt|  channelTitle|categoryId|trending_date|view_count|   likes|dislikes|comment_count|publishedDate|daystakentotrend|
    +-----------+--------------------+-------------------+--------------+----------+-------------+----------+--------+--------+-------------+-------------+----------------+
    |3JxF7DuJzjc|ROCKETS at THUNDE...|2020-09-01 03:58:10|           NBA|        17|   2020-09-01| 1174561.0| 10178.0|   478.0|       2768.0|   2020-09-01|               0|
    |bWTEmyTpBrQ|You can change yo...|2021-02-11 01:01:35|      Flamingo|        20|   2021-02-11| 1351061.0| 88299.0|   611.0|      18862.0|   2021-02-11|               0|
    |Rh_mIhLpAnA|Value Pack Games ...|2021-03-29 00:00:03| Scott The Woz|        20|   2021-03-29|  600735.0| 61256.0|   303.0|       4789.0|   2021-03-29|               0|
    |0eF0Ax8aGfc|Toosii - shop (Of...|2021-05-07 04:02:25|  Toosii2xVEVO|        10|   2021-05-07|  389841.0| 34250.0|   342.0|       4417.0|   2021-05-07|               0|
    |9syCTdjUhCA|Olivia Rodrigo - ...|2021-06-30 04:06:05|Olivia Rodrigo|        10|   2021-06-30| 3243728.0|537634.0|  1567.0|      17988.0|   2021-06-30|               0|
    +-----------+--------------------+-------------------+--------------+----------+-------------+----------+--------+--------+-------------+-------------+----------------+
    only showing top 5 rows

    ====== Trending videos from Category : People & Blogs took longer time to trend compared to other categories. ======
    +---------------------+----------+--------------------+
    |Category             |categoryId|avg_daystakentotrend|
    +---------------------+----------+--------------------+
    |People & Blogs       |22        |1.6757337151037939  |
    |Pets & Animals       |15        |1.5730337078651686  |
    |Comedy               |23        |1.5206422018348624  |
    |Entertainment        |24        |1.5093599033816425  |
    |News & Politics      |25        |1.4807987711213517  |
    |Gaming               |20        |1.4688156972669937  |
    |Howto & Style        |26        |1.4557692307692307  |
    |Film & Animation     |1         |1.451559934318555   |
    |Science & Technology |28        |1.4078303425774878  |
    |Education            |27        |1.3948126801152738  |
    |Autos & Vehicles     |2         |1.3863636363636365  |
    |Music                |10        |1.3696160267111852  |
    |Nonprofits & Activism|29        |1.3333333333333333  |
    |Sports               |17        |1.3162970106075218  |
    |Travel & Events      |19        |1.289855072463768   |
    +---------------------+----------+--------------------+

    time: 59.7 s (started: 2021-12-10 17:32:46 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="UyjPKdyQDqyi" outputId="1ae4fa1c-4501-4b1e-c613-db2cfa6638c2"}
``` {.python}
# b) How long does it take a Video to become trending for the first time? Videos from which category has a longer trend? 
inputdf_1b = finaldf_1a.withColumn("publishedDate", expr("date(publishedAt)")).withColumn("timetakentotrend", expr("trending_date - publishedDate"))\
             .where(col("timetakentotrend").isNotNull()).withColumn("timetakentotrend",col("timetakentotrend").cast(StringType())).drop("channelId","comments_disabled","ratings_disabled")

split_timetaken_1b = F.split(inputdf_1b['timetakentotrend'], ' ')

daystakentotrenddf_1b = inputdf_1b.withColumn("daystakentotrend", split_timetaken_1b.getItem(0))\
            .withColumn("daystakentotrend",col("daystakentotrend").cast(IntegerType())).drop("timetakentotrend", "publishedAt")

maxdaystotrenddf_1b = daystakentotrenddf_1b.orderBy("daystakentotrend", ascending=False)
## Maximum it took 27 days for a video to trend from its publish date. (3 videos : All from Walmart)
print("====== Maximum it took "+ str(maxdaystotrenddf_1b.collect()[0][10]) + " days for a video to trend from its publish date. ======")
maxdaystotrenddf_1b.show(4, truncate=False)

## 1087 videos Started trending on same day as Published.
zerodaystotrenddf_1b = daystakentotrenddf_1b.where(col("daystakentotrend") == 0)
print("====== "+ str(zerodaystotrenddf_1b.count()) + " videos Started trending on same day as Published. ======")
zerodaystotrenddf_1b.show(5)
#zerodaystotrenddf_1b.count()

## Trending videos from Category ID 22 ("People & Blogs") took longer time to trend compared to other categories.
longertrendbycatdf_1b = daystakentotrenddf_1b.groupBy("categoryId").agg(F.avg("daystakentotrend"))\
                        .withColumnRenamed("avg(daystakentotrend)", "avg_daystakentotrend").orderBy("avg_daystakentotrend", ascending=False)

# Add Category Name from JASON File
categorynamedf_1b = ytvusjdf.join(longertrendbycatdf_1b, (longertrendbycatdf_1b.categoryId == ytvusjdf.id))\
                    .drop("id").orderBy("avg_daystakentotrend", ascending=False)
print("====== Trending videos from Category : "+ str(categorynamedf_1b.collect()[0][0]) + " took longer time to trend compared to other categories. ======")
categorynamedf_1b.show(20, truncate=False)
```

::: {.output .stream .stdout}
    ====== Maximum it took 27 days for a video to trend from its publish date. ======
    +-----------+-----------------------------------------------------------+------------+----------+-------------+-----------+-------+--------+-------------+-------------+----------------+
    |video_id   |title                                                      |channelTitle|categoryId|trending_date|view_count |likes  |dislikes|comment_count|publishedDate|daystakentotrend|
    +-----------+-----------------------------------------------------------+------------+----------+-------------+-----------+-------+--------+-------------+-------------+----------------+
    |TR0I0STLyws|A different kind of membership. Walmart+ | Mobile scan & go|Walmart     |22        |2020-10-12   |117988.0   |410.0  |283.0   |0.0          |2020-09-15   |27              |
    |izVdUXx5zaU|Walmart Back to College | However you study                |Walmart     |22        |2020-09-02   |1205139.0  |24.0   |19.0    |16.0         |2020-08-06   |27              |
    |MeYkO5vkakg|A different kind of membership. Walmart+ | The Cox Family  |Walmart     |22        |2020-10-12   |1.1624348E7|9273.0 |5636.0  |0.0          |2020-09-15   |27              |
    |Zsfo9ECSEtA|how to open packaged scissors with no scissors             |Alex Martens|23        |2020-11-08   |467864.0   |14478.0|968.0   |629.0        |2020-10-13   |26              |
    +-----------+-----------------------------------------------------------+------------+----------+-------------+-----------+-------+--------+-------------+-------------+----------------+
    only showing top 4 rows

    ====== 1092 videos Started trending on same day as Published. ======
    +-----------+--------------------+--------------+----------+-------------+----------+--------+--------+-------------+-------------+----------------+
    |   video_id|               title|  channelTitle|categoryId|trending_date|view_count|   likes|dislikes|comment_count|publishedDate|daystakentotrend|
    +-----------+--------------------+--------------+----------+-------------+----------+--------+--------+-------------+-------------+----------------+
    |3JxF7DuJzjc|ROCKETS at THUNDE...|           NBA|        17|   2020-09-01| 1174561.0| 10178.0|   478.0|       2768.0|   2020-09-01|               0|
    |bWTEmyTpBrQ|You can change yo...|      Flamingo|        20|   2021-02-11| 1351061.0| 88299.0|   611.0|      18862.0|   2021-02-11|               0|
    |Rh_mIhLpAnA|Value Pack Games ...| Scott The Woz|        20|   2021-03-29|  600735.0| 61256.0|   303.0|       4789.0|   2021-03-29|               0|
    |0eF0Ax8aGfc|Toosii - shop (Of...|  Toosii2xVEVO|        10|   2021-05-07|  389841.0| 34250.0|   342.0|       4417.0|   2021-05-07|               0|
    |9syCTdjUhCA|Olivia Rodrigo - ...|Olivia Rodrigo|        10|   2021-06-30| 3243728.0|537634.0|  1567.0|      17988.0|   2021-06-30|               0|
    +-----------+--------------------+--------------+----------+-------------+----------+--------+--------+-------------+-------------+----------------+
    only showing top 5 rows

    ====== Trending videos from Category : People & Blogs took longer time to trend compared to other categories. ======
    +---------------------+----------+--------------------+
    |Category             |categoryId|avg_daystakentotrend|
    +---------------------+----------+--------------------+
    |People & Blogs       |22        |1.6757337151037939  |
    |Pets & Animals       |15        |1.5730337078651686  |
    |Comedy               |23        |1.5206422018348624  |
    |Entertainment        |24        |1.5093599033816425  |
    |News & Politics      |25        |1.4807987711213517  |
    |Gaming               |20        |1.4688156972669937  |
    |Howto & Style        |26        |1.4557692307692307  |
    |Film & Animation     |1         |1.451559934318555   |
    |Science & Technology |28        |1.4078303425774878  |
    |Education            |27        |1.3948126801152738  |
    |Autos & Vehicles     |2         |1.3863636363636365  |
    |Music                |10        |1.3696160267111852  |
    |Nonprofits & Activism|29        |1.3333333333333333  |
    |Sports               |17        |1.3162970106075218  |
    |Travel & Events      |19        |1.289855072463768   |
    +---------------------+----------+--------------------+

    time: 57.7 s (started: 2021-12-10 17:33:46 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":812,\"base_uri\":\"https://localhost:8080/\"}" id="cSBx_7R2MS3d" outputId="67fbf897-3799-455e-a673-f3868c579ee9"}
``` {.python}
#Histogram for X: No of Days Taken to Trend (Vs) Y: number of videos per bin/bucket.
VDFHIST1DAYS_1B = daystakentotrenddf_1b.select("daystakentotrend").orderBy("daystakentotrend", ascending=False)
VDFHIST1DAYS_1BOUT = [data[0] for data in VDFHIST1DAYS_1B.select('daystakentotrend').collect()]

fig, ax = plt.subplots(figsize=(15,10))
ax.hist(VDFHIST1DAYS_1BOUT, bins = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28])
ax.set_title("Days Taken to Trend (Vs) Number of videos")
ax.set_xlabel('No. of Days Taken to Trend')
ax.set_ylabel('Number of videos')
ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28])
plt.yscale('log')
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/81528535e46eee4f232e2ae1a71e6f6ddd2c1587.png)
:::

::: {.output .stream .stdout}
    time: 11.1 s (started: 2021-12-10 17:34:44 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":715,\"base_uri\":\"https://localhost:8080/\"}" id="LdjKp8RMmE2W" outputId="32813e1c-cc6e-4fc4-c2b4-8fe613bdc7ce"}
``` {.python}
# Bar Plot for 1B : Category (Vs) Avg. Days taken to trend
import numpy as np
import matplotlib.pyplot as plt

VDFBAR1_1B = categorynamedf_1b.select("Category")
VDFBAR1_1BOUT = [data[0] for data in VDFBAR1_1B.select('Category').collect()]
#print(VDFBAR1_1BOUT)

VDFBAR2_1B = categorynamedf_1b.select("avg_daystakentotrend")
VDFBAR2_1BOUT = [data[0] for data in VDFBAR2_1B.select('avg_daystakentotrend').collect()]
#print(VDFBAR2_1BOUT)

fig, ax = plt.subplots(figsize=(15,10))
#ax.bar(VDFBAR1_1BOUT,VDFBAR2_1BOUT)
sns.set_style("whitegrid")
ax = sns.barplot(y=VDFBAR1_1BOUT,x=VDFBAR2_1BOUT,orient="h")
ax.set_title("Category (Vs) Avg. Days taken to trend")
ax.set_xlabel('Avg. Days taken to trend')
ax.set_ylabel('Category')
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/6dd4946de0ab055a81c33e27cf5c2355ddaa6aae.png)
:::

::: {.output .stream .stdout}
    time: 24.6 s (started: 2021-12-10 17:34:55 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="frnNJ1xrlIdH" outputId="41ad8478-49c1-4326-8765-f7236e00c41a"}
``` {.python}
# c) What percentage of trending Videos have more Dislikes than Likes? (Negative publicity inference) 

# On 16455 First time trending videos 
popularbutnotlikeddf_1c = finaldf_1a.where(expr("dislikes - likes") > 0)
## 59 / 16455 = Only 0.36 % Videos have more Dislikes than likes when they first became trending.
print("====== "+ str(popularbutnotlikeddf_1c.count()) + " Videos out of "+ str(finaldf_1a.count()) +" i.e. Only 0.36% Videos have more Dislikes than likes when they first became trending. ======")
popularbutnotlikeddf_1c.count() # 59
# finaldf_1a.count() # 16455


# On all 91791 videos 
## 363/91791 = 0.395 of total Videos have more Dislikes than likes.
totalnotlikeddf_1c = cleandf_1a.where(expr("dislikes - likes") > 0)
print("====== "+ str(totalnotlikeddf_1c.count()) +" Videos out of "+ str(cleandf_1a.count()) + " i.e. Only 0.395% of total Videos have more Dislikes than likes. ======")
totalnotlikeddf_1c.count() # 363
# cleandf_1a.count() # 91791
```

::: {.output .stream .stdout}
    ====== 60 Videos out of 16728 i.e. Only 0.36% Videos have more Dislikes than likes when they first became trending. ======
    ====== 363 Videos out of 91791 i.e. Only 0.395% of total Videos have more Dislikes than likes. ======
:::

::: {.output .execute_result execution_count="65"}
    363
:::

::: {.output .stream .stdout}
    time: 28.4 s (started: 2021-12-10 17:35:19 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="hA4fr_TQlKsX" outputId="d7e99681-9c40-4af4-80ab-571380109c4c"}
``` {.python}
# d) Which category of Video becomes most trending? Users like videos from which CATEGORY the most? 

# Which category of Video becomes most trending?
categorydf_1d = cleandf_1a.select("categoryId").withColumn("count", expr("1")).groupBy("categoryId").agg(F.sum("count"))\
                 .withColumnRenamed("sum(count)","count")

# Add Category Name from JASON File
categorynamedf_1d = ytvusjdf.join(categorydf_1d, (categorydf_1d.categoryId == ytvusjdf.id))\
                    .drop("id").orderBy("count", ascending=False)

## Ofcourse "Entertainment" Category (24) becomes more trending with 18373 trending videos count
## "Nonprofits & Activism" Category (29) stands least with 88 trending videos count
cat1dcount = categorynamedf_1d.count()
print("====== Ofcourse "+ str(categorynamedf_1d.collect()[0][0]) + " Category (24) becomes more trending with "+ str(categorynamedf_1d.collect()[0][2]) +" trending videos count ======")
print("====== "+str(categorynamedf_1d.collect()[cat1dcount-1][0]) + " Category (29) stands least with "+ str(categorynamedf_1d.collect()[cat1dcount-1][2]) +" trending videos count ======")
categorynamedf_1d.show(cat1dcount, truncate=False)
```

::: {.output .stream .stdout}
    ====== Ofcourse Entertainment Category (24) becomes more trending with 18373 trending videos count ======
    ====== Nonprofits & Activism Category (29) stands least with 88 trending videos count ======
    +---------------------+----------+-----+
    |Category             |categoryId|count|
    +---------------------+----------+-----+
    |Entertainment        |24        |18373|
    |Music                |10        |17053|
    |Gaming               |20        |15489|
    |Sports               |17        |10093|
    |People & Blogs       |22        |7845 |
    |Comedy               |23        |4997 |
    |Film & Animation     |1         |3511 |
    |News & Politics      |25        |3400 |
    |Science & Technology |28        |3371 |
    |Howto & Style        |26        |3039 |
    |Education            |27        |2003 |
    |Autos & Vehicles     |2         |1650 |
    |Pets & Animals       |15        |494  |
    |Travel & Events      |19        |385  |
    |Nonprofits & Activism|29        |88   |
    +---------------------+----------+-----+

    time: 7.73 s (started: 2021-12-10 17:35:48 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":734,\"base_uri\":\"https://localhost:8080/\"}" id="_lHCT_x5mxp8" outputId="116a5cde-5aa8-4fee-e4f4-728974d5b188"}
``` {.python}
##Plot for Which category of Video becomes most trending?

VDFBAR1_1D = categorynamedf_1d.select("Category")
VDFBAR1_1DOUT = [data[0] for data in VDFBAR1_1D.select('Category').collect()]
#print(VDFBAR1_1BOUT)

VDFBAR2_1D = categorynamedf_1d.select("count")
VDFBAR2_1DOUT = [data[0] for data in VDFBAR2_1D.select('count').collect()]
#print(VDFBAR2_1BOUT)

plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=VDFBAR1_1DOUT,x=VDFBAR2_1DOUT,orient="h")
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Catogories of trend videos in US")
```

::: {.output .execute_result execution_count="67"}
    Text(0.5, 1.0, 'Catogories of trend videos in US')
:::

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/1eb4d12581d4d9a80dbe9b0622b0f2289ca7bcb9.png)
:::

::: {.output .stream .stdout}
    time: 6.66 s (started: 2021-12-10 17:35:56 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="fYv14tCfX9kD" outputId="9ed0b6b2-a5f5-4bee-a931-3af3ae9446e3"}
``` {.python}
#Users like videos from which CATEGORY the most?

likesbyCatergorydf_1d = cleandf_1a.select("categoryId", "likes").groupBy("categoryId").agg(F.avg("likes"))\
                        .withColumnRenamed("avg(likes)","Avg_likes")
likesbyCatergorynamedf_1d = ytvusjdf.join(likesbyCatergorydf_1d, (likesbyCatergorydf_1d.categoryId == ytvusjdf.id))\
                           .drop("id").orderBy("Avg_likes", ascending=False)
## "Music" Category(10) has highest avg of likes (~324052) and "News & Politics" with least avg of likes (22934)
cat1dlikescount = likesbyCatergorynamedf_1d.count()
print("====== "+ str(likesbyCatergorynamedf_1d.collect()[0][0]) + " Category("+ str(likesbyCatergorynamedf_1d.collect()[0][1]) +") has highest avg of likes ======")
print("====== "+ str(likesbyCatergorynamedf_1d.collect()[cat1dlikescount-1][0]) + " Category("+ str(likesbyCatergorynamedf_1d.collect()[cat1dlikescount-1][1]) +") with least avg of likes ======")
likesbyCatergorynamedf_1d.show(cat1dlikescount, truncate=False)

dislikesbyCatergorydf_1d = cleandf_1a.select("categoryId", "dislikes").groupBy("categoryId").agg(F.avg("dislikes"))\
                        .withColumnRenamed("avg(dislikes)","dislikes")
dislikesbyCatergorynamedf_1d = ytvusjdf.join(dislikesbyCatergorydf_1d, (dislikesbyCatergorydf_1d.categoryId == ytvusjdf.id))\
                           .drop("id").orderBy("dislikes", ascending=False)
cat1ddislikescount = likesbyCatergorynamedf_1d.count()
## "Music" Category(10) has highest avg of dislikes also (~6068) and "Pets & Animals" with least avg of dislikes (743)
print("====== "+ str(dislikesbyCatergorynamedf_1d.collect()[0][0]) + " Category("+ str(dislikesbyCatergorynamedf_1d.collect()[0][1]) +") has highest avg of dislikes also ======")
print("====== "+ str(dislikesbyCatergorynamedf_1d.collect()[cat1ddislikescount-1][0]) + " Category("+ str(dislikesbyCatergorynamedf_1d.collect()[cat1ddislikescount-1][1]) +") with least avg of dislikes ======")
dislikesbyCatergorynamedf_1d.show(cat1ddislikescount, truncate=False)
```

::: {.output .stream .stdout}
    ====== Music Category(10) has highest avg of likes ======
    ====== News & Politics Category(25) with least avg of likes ======
    +---------------------+----------+------------------+
    |Category             |categoryId|Avg_likes         |
    +---------------------+----------+------------------+
    |Music                |10        |324052.5737406908 |
    |Entertainment        |24        |167267.7152343112 |
    |People & Blogs       |22        |132811.45086042065|
    |Gaming               |20        |128413.86868099943|
    |Nonprofits & Activism|29        |126097.29545454546|
    |Comedy               |23        |122843.45827496499|
    |Science & Technology |28        |115669.24592109167|
    |Film & Animation     |1         |100102.31643406437|
    |Education            |27        |95534.7668497254  |
    |Howto & Style        |26        |70824.82856202699 |
    |Pets & Animals       |15        |58516.96963562753 |
    |Sports               |17        |51975.609531358365|
    |Autos & Vehicles     |2         |51397.65393939394 |
    |Travel & Events      |19        |46667.55324675325 |
    |News & Politics      |25        |22934.119705882353|
    +---------------------+----------+------------------+

    ====== Music Category(10) has highest avg of dislikes also ======
    ====== Pets & Animals Category(15) with least avg of dislikes ======
    +---------------------+----------+------------------+
    |Category             |categoryId|dislikes          |
    +---------------------+----------+------------------+
    |Music                |10        |6068.090423972321 |
    |People & Blogs       |22        |3782.9287444231995|
    |Entertainment        |24        |3222.7866434441844|
    |News & Politics      |25        |2879.3573529411765|
    |Science & Technology |28        |2685.7214476416493|
    |Gaming               |20        |2350.9191684421203|
    |Howto & Style        |26        |2320.081605791379 |
    |Comedy               |23        |1902.2513508104862|
    |Film & Animation     |1         |1795.5548276844204|
    |Education            |27        |1667.7069395906142|
    |Sports               |17        |1500.38373129892  |
    |Nonprofits & Activism|29        |1283.7272727272727|
    |Autos & Vehicles     |2         |808.9575757575758 |
    |Travel & Events      |19        |780.2961038961039 |
    |Pets & Animals       |15        |743.1963562753036 |
    +---------------------+----------+------------------+

    time: 15.5 s (started: 2021-12-10 17:36:02 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":734,\"base_uri\":\"https://localhost:8080/\"}" id="c4SPBYkTdHi1" outputId="bfc2f03c-f9eb-4ca1-a7e2-a455d5aabc01"}
``` {.python}
##Plot for Users like videos from which CATEGORY the most?
import seaborn as sns

VDFBAR3_1D = likesbyCatergorynamedf_1d.select("Category")
VDFBAR3_1DOUT = [data[0] for data in VDFBAR3_1D.select('Category').collect()]
#print(VDFBAR1_1BOUT)

VDFBAR4_1D = likesbyCatergorynamedf_1d.select("Avg_likes")
VDFBAR4_1DOUT = [data[0] for data in VDFBAR4_1D.select('Avg_likes').collect()]
#print(VDFBAR2_1BOUT)

plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=VDFBAR3_1DOUT,x=VDFBAR4_1DOUT,orient="h")
plt.xlabel("Avg. Number of Likes")
plt.ylabel("Categories")
plt.title("Catogories Vs Likes of trend videos in US")
```

::: {.output .execute_result execution_count="69"}
    Text(0.5, 1.0, 'Catogories Vs Likes of trend videos in US')
:::

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/87f98f58b745fce2a5521918d92cad8972907ae9.png)
:::

::: {.output .stream .stdout}
    time: 6.52 s (started: 2021-12-10 17:36:18 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="e58UMXd5lM6S" outputId="1ec8dd61-7884-40d9-dd1f-89040b6ff86e"}
``` {.python}
# e) Which channels produced more trending videos? 

Channelvidcountdf_1e = cleandf_1a.select("channelId", "channelTitle").withColumn("count", expr("1"))\
                      .groupBy("channelId", "channelTitle").agg(F.sum("count")).withColumnRenamed("channelId","channel_Id")\
                      .withColumnRenamed("sum(count)","count").orderBy("count", ascending=False)

## "NBA" channel (with Channel ID: UCWJ2lWNubArHWmf3FIHbfcQ) has hosted highest trending videos (578).
channel1ecount = Channelvidcountdf_1e.count()
print("====== "+ str(Channelvidcountdf_1e.collect()[0][1]) + " channel(with Channel ID:"+ str(Channelvidcountdf_1e.collect()[0][0]) +") has hosted highest trending videos " + str(Channelvidcountdf_1e.collect()[0][2]) +" ======")
Channelvidcountdf_1e.show(2, truncate=False)

## 41 channels has only 1 video trending and only on one day.
Channelonevideodf_1e = Channelvidcountdf_1e.filter(col("count") == "1")
print("====== "+ str(Channelonevideodf_1e.count()) + " channels has only "+ str(Channelonevideodf_1e.collect()[0][2]) +" video trending and only on one day. ======")
Channelonevideodf_1e.show(2, truncate=False)
```

::: {.output .stream .stdout}
    ====== NBA channel(with Channel ID:UCWJ2lWNubArHWmf3FIHbfcQ) has hosted highest trending videos 578 ======
    +------------------------+------------+-----+
    |channel_Id              |channelTitle|count|
    +------------------------+------------+-----+
    |UCWJ2lWNubArHWmf3FIHbfcQ|NBA         |578  |
    |UCDVYQ4Zhbm3S2dlz7P1GBDg|NFL         |572  |
    +------------------------+------------+-----+
    only showing top 2 rows

    ====== 41 channels has only 1 video trending and only on one day. ======
    +------------------------+-------------------+-----+
    |channel_Id              |channelTitle       |count|
    +------------------------+-------------------+-----+
    |UC1JOnWZrVWKzX3UMdpnvuMg|Paradox Interactive|1    |
    |UCfrSUhUYOUyZROqWZb-TphQ|adidas Originals   |1    |
    +------------------------+-------------------+-----+
    only showing top 2 rows

    time: 17.2 s (started: 2021-12-10 17:36:24 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="cY-0weewmCOR" outputId="a30dc133-72ab-48f8-8493-eb76463e95b4"}
``` {.python}
# f) How many videos appeared trending for most of the days?

## Correlation between Days of Publish to Trend (v/s) Trending Duration.

df3 = usadf.where(col("Description").isNotNull()).withColumn("publishedAt",F.to_timestamp(col("publishedAt"))).\
withColumn("trending_date",F.to_timestamp(col("trending_date"))).\
withColumn("Date Diff",F.datediff(col("trending_date"),col("publishedAt"))).\
groupby(col("video_id")).agg(F.min("Date Diff"),F.count("Date Diff")).\
withColumnRenamed("min(Date Diff)","Days to Trend").\
withColumnRenamed("count(Date Diff)","Trending Duration")
```

::: {.output .stream .stdout}
    time: 76.3 ms (started: 2021-12-10 17:36:42 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":504,\"base_uri\":\"https://localhost:8080/\"}" id="ZB7CgltH3aw2" outputId="e0e2cba4-6d32-47a1-9dae-b5e4174c8cec"}
``` {.python}
## Plot for 1f : How many videos appeared trending for most of the days?
trendingHist = [data[0] for data in df3.select('Trending Duration').collect()]

from matplotlib import pyplot as plt
import numpy as np
fig,ax = plt.subplots(1,1)

ax.hist(trendingHist, bins = [0,2,4,6,8,10,12,14,16,18,20])
ax.set_title("Histogram of Trending Duration")
ax.set_xlabel('Trending duration (Days)')
ax.set_ylabel('Number of Videos')
ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/9f99a643871c63789c756aadcb4e0a79f5306861.png)
:::

::: {.output .stream .stdout}
    time: 3.42 s (started: 2021-12-10 17:36:42 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="-vfO1D0q4J2v" outputId="490296e0-31ae-4366-de1e-e6314b1041ca"}
``` {.python}
# Section 2 -  Starts Here
```

::: {.output .stream .stdout}
    time: 985 µs (started: 2021-12-10 17:36:45 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="datuqFrT4ECn" outputId="0dd9d7f5-afa9-48d3-a215-15b5dbaaadf2"}
``` {.python}
# Section 3  - Starts Here
```

::: {.output .stream .stdout}
    time: 1.02 ms (started: 2021-12-10 17:36:45 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="7MEMvccAKBZo" outputId="0d328af8-dc37-44d2-d4a3-85d2fb450415"}
``` {.python}
### Helper function for drawing heatmap
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)
    

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
```

::: {.output .stream .stdout}
    time: 73.9 ms (started: 2021-12-10 17:36:45 +00:00)
:::
:::

::: {.cell .markdown id="7JrzRvx2AQ-4"}
#### Q3A Find correlation ration between likes-dislikes-view-comment for different categories. {#q3a-find-correlation-ration-between-likes-dislikes-view-comment-for-different-categories}
:::

::: {.cell .code colab="{\"height\":837,\"base_uri\":\"https://localhost:8080/\"}" id="k22_e2vWAOD8" outputId="f87593eb-2655-45c7-e270-200a98f3f0b3"}
``` {.python}

df11 = usadf.withColumn("view_count",col("view_count").cast("double"))\
.withColumn("likes",col("likes").cast("double"))\
.withColumn("dislikes",col("dislikes").cast("double"))\
.withColumn("comment_count",col("comment_count").cast("double"))\
.withColumn("categoryId",F.col("categoryId").cast("int"))

df22 = df11.groupby('categoryId').sum().withColumn("ratioLikeDisLike",col('sum(likes)')/col('sum(dislikes)')).\
withColumn("ratioViewComment",col('sum(view_count)')/col('sum(comment_count)')).\
withColumn("ratioDisLikeView",col('sum(dislikes)')/col('sum(view_count)')).\
withColumn("ratioLikeView",col('sum(likes)')/col('sum(view_count)'))

df33 = df22.join(ytvusjdf,df22.categoryId == ytvusjdf.id)

def plotData(data_x,data_y,title):
  sns.set_style("whitegrid")
  sns.set(font_scale = 2)
  sns.barplot(x = data_x, y = data_y, orient='h')
  plt.xlabel("Ratio")
  plt.ylabel("Category")
  plt.title(title)

plt.figure(figsize=(25,20))
plt.subplot(231)
x = df33.select(["ratioLikeDisLike","Category"]).sort(["ratioLikeDisLike"],ascending=False).collect()
data_x = [item['ratioLikeDisLike'] for item in x]; data_y = [item['Category'] for item in x]
plotData(data_x,data_y,"Like - Dislike Ratio")

plt.subplot(233)
x = df33.select(["ratioViewComment","Category"]).sort(["ratioViewComment"],ascending=False).collect()
data_x = [item['ratioViewComment'] for item in x]; data_y = [item['Category'] for item in x]
plotData(data_x,data_y, "View - Comment Ratio")

plt.subplot(234)
x = df33.select(["ratioDisLikeView","Category"]).sort(["ratioDisLikeView"],ascending=False).collect()
data_x = [item['ratioDisLikeView'] for item in x]; data_y = [item['Category'] for item in x]
plotData(data_x,data_y,"Dislike - View Ratio")

plt.subplot(236)
x = df33.select(["ratioLikeView","Category"]).sort(["ratioLikeView"],ascending=False).collect()
data_x = [item['ratioLikeView'] for item in x]; data_y = [item['Category'] for item in x]
plotData(data_x,data_y, "Like - View Ratio")

plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/7a4d2f41c8f2a37f10b4eaa05e839a050edb9ebe.png)
:::

::: {.output .stream .stdout}
    time: 15.9 s (started: 2021-12-10 17:36:45 +00:00)
:::
:::

::: {.cell .markdown id="RTy2yVWxAUyF"}
Observations:

1.  News & Politics have lowest like-dislike ratio and view-comment
    ratio. People relatively dislike these videos and comment alot. It
    also have highest Dislike- view ratio.

2.  Pets & Animals videos & Non-profits Videos have highest
    likes-dislikes ratio. Not suprisingly, people find difficult to hate
    pets and animals and non-profit activity.

3.  Sports and Science Technology have largest view-comment ratio.
    People tend to comment less on these videos compared to
    music-related video.
:::

::: {.cell .markdown id="5jOiku94jZc0"}
Q 3.B **Correlation between different metrics through Heat map.**
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="I33YjIBORF-q" outputId="9e4256a2-aa36-410d-830b-8be18eec4da2"}
``` {.python}
### Get Features
df = usadf.where(col("description").isNotNull()).withColumn("Description length",length("description"))\
.withColumn("Title length",length("title"))\
.withColumn("Tags length",length("tags"))\
.withColumn("# of Tags",size(split(col("tags"),"[|]")))\
.select("view_count","likes","dislikes","comment_count","Description length","Title length","Tags length","# of Tags","categoryId")

### Format data
df1 = df.withColumn("view_count",col("view_count").cast("double"))\
.withColumn("likes",col("likes").cast("double"))\
.withColumn("dislikes",col("dislikes").cast("double"))\
.withColumn("comment_count",col("comment_count").cast("double"))\
.withColumn("categoryId",F.col("categoryId").cast("int"))

df2 = df1.where(col("categoryId").isNotNull())
```

::: {.output .stream .stdout}
    time: 181 ms (started: 2021-12-10 17:37:01 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ZzB4l57PAkxe" outputId="6aef8859-6eb1-4b59-c2a0-115578970414"}
``` {.python}
categories = [24,29]

## Convert dataframe into vectors
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df2.columns[:-1], outputCol=vector_col)
  
### Find correlation of every category for the columns specified and store it in dictionary
corr_list = []
for ID in categories:
  # convert to vector column first
  df_vector = assembler.transform(df2.where(col("categoryId") == ID)).select(vector_col)
  # get correlation matrix
  matrix = Correlation.corr(df_vector, vector_col)
  name = ytvusjdf.where(col("id") == ID).select("Category").collect()[0]['Category']
  corr_list.append({"ID":ID,"name":name,"Corr":matrix.collect()[0]['pearson(corr_features)'].values})
```

::: {.output .stream .stdout}
    time: 7.26 s (started: 2021-12-10 17:37:01 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":435,\"base_uri\":\"https://localhost:8080/\"}" id="n1HG2dm2VCLY" outputId="baef626b-6fc7-417d-a861-c509b6a3cbe8"}
``` {.python}
## Plot the data
sns.reset_defaults()

columns = ["view_count","likes","dislikes","comment_count","Description length","Title length","Tags length","# of Tags"]
plt.figure(figsize=(20,6))
plt.subplot(121)
im, cbar = heatmap(corr_list[0]["Corr"].reshape(len(columns),-1), columns, columns, ax=None, cmap="Blues", cbarlabel=corr_list[0]["name"])
texts = annotate_heatmap(im, valfmt="{x:.2f}")

plt.subplot(122)
im, cbar = heatmap(corr_list[1]["Corr"].reshape(len(columns),-1), columns, columns, ax=None,cmap="Blues", cbarlabel=corr_list[1]["name"])
texts = annotate_heatmap(im, valfmt="{x:.2f}")

plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/2649697cc07c89cdce87de0a0a1cdf8ee65828ad.png)
:::

::: {.output .stream .stdout}
    time: 1.71 s (started: 2021-12-10 17:37:09 +00:00)
:::
:::

::: {.cell .markdown id="J82ETMKiDPPy"}
Observations:

1.  Videos belong to entertainment category shows high correlation among
    views, likes, dislikes etc. It shows very less correlation with
    description length, title length etc

2.  However, for video which are trending on category like Non-profits
    and activism, Description length plays an important role.

3.  49% correlation is observed between views and description length and
    84% correlation between dislikes and description length.
:::

::: {.cell .markdown id="Bi_-Bm3MEBpW"}
## Q-3c Correlation between Days of Publish to Trend (v/s) Trending Duration. {#q-3c-correlation-between-days-of-publish-to-trend-vs-trending-duration}
:::

::: {.cell .code colab="{\"height\":582,\"base_uri\":\"https://localhost:8080/\"}" id="csHx_6fBhpyt" outputId="868c3ab3-f6aa-4548-c073-2826985659d0"}
``` {.python}
## Get features
df3 = usadf.where(col("Description").isNotNull()).withColumn("publishedAt",F.to_timestamp(col("publishedAt"))).\
withColumn("trending_date",F.to_timestamp(col("trending_date"))).\
withColumn("Date Diff",F.datediff(col("trending_date"),col("publishedAt"))).\
groupby(col("video_id")).agg(F.min("Date Diff"),F.count("Date Diff")).\
withColumnRenamed("min(Date Diff)","Days to Trend").\
withColumnRenamed("count(Date Diff)","Trending Duration")

## Get count
out = df3.groupby("Days to Trend","Trending Duration").count().collect()

## Convert to heatmap array
out1 = np.array([(item['Days to Trend'], item['Trending Duration'], item['count']) for item in out])
out2 = np.zeros((np.max(out1[:,0])+1, np.max(out1[:,1]) + 1)) * np.nan
out2[out1[:,0],out1[:,1]] = out1[:,2]

## Plot data
plt.figure(figsize=(10,6))
ax = sns.heatmap(out2.T, cmap='viridis_r')
ax.invert_yaxis()
plt.title("Correlation between Days to Publish v/s Trend and Trending Duration")
plt.xlabel("Days to trend")
plt.ylabel("Trending Duration")
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/93b4415201cdbcb3f99b918cda27c9ae68bae029.png)
:::

::: {.output .stream .stdout}
    time: 6.71 s (started: 2021-12-10 17:37:10 +00:00)
:::
:::

::: {.cell .markdown id="4k-T7f0JChyv"}
Observations:

1.  The less days needed for a video from publish to trend, the longer
    the trend duration.

2.  Videos that can get into trending within 3 days will have higher
    probability to be trending for longer time.
:::

::: {.cell .markdown id="3XhgszYV4ckk"}
#### 3. Correlation between various trending video metrics: \[Spark Dataframes\] {#3-correlation-between-various-trending-video-metrics-spark-dataframes}

\#\#\#\#\# d) Does the video trending in one country will trend in other
countries too? (for 3 countries) 1) get the distinct trending video per
country.

2\) Count the number of trending video matching between US vs GB and US
vs IN

3\) Derive the analogy.
:::

::: {.cell .code colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="JDoHyhaj41Oh" outputId="36c25b0d-8a49-4394-d0a6-b378fcc3a9fa"}
``` {.python}
usDistinctdf = uscdf.select("video_id").distinct()
gbDistinctdf = gbcdf.select("video_id").distinct().withColumnRenamed("video_id","gbvideo_id")
inDistinctdf = incdf.select("video_id").distinct().withColumnRenamed("video_id","invideo_id")

usGbMatchdf = usDistinctdf.join(gbDistinctdf, usDistinctdf.video_id == gbDistinctdf.gbvideo_id).drop("gbvideo_id")
usInMatchdf = usDistinctdf.join(inDistinctdf, usDistinctdf.video_id == inDistinctdf.invideo_id).drop("invideo_id")
gbInMatchdf = gbDistinctdf.join(inDistinctdf, gbDistinctdf.gbvideo_id == inDistinctdf.invideo_id).drop("gbvideo_id")
all3Matchdf = usGbMatchdf.join(inDistinctdf, usGbMatchdf.video_id == inDistinctdf.invideo_id).drop("invideo_id")

usCount = usDistinctdf.count()
gbCount = gbDistinctdf.count()
inCount = inDistinctdf.count()
usGbCount = usGbMatchdf.count()
usInCount = usInMatchdf.count()
gbInCount = gbInMatchdf.count()
allCount = all3Matchdf.count()

fig, ax = plt.subplots(2, 2, figsize=(18, 18), subplot_kw=dict(aspect="equal"))

allCountrydata = [allCount, usCount-allCount, gbCount-allCount, inCount-allCount]
ingredients = ["COMMON", "USA ONLY", "GRATE BRITAIN ONLY", "INDIA ONLY"]

def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d} Count)".format(pct, absolute)

wedges, texts, autotexts = ax[0][0].pie(allCountrydata, autopct=lambda pct: func(pct, allCountrydata), textprops=dict(color="w"))
ax[0][0].legend(wedges, ingredients, title="COUNTRY", loc="center", bbox_to_anchor=(1, 0.8, 0, 0))
ax[0][0].set_title("All Country Trending Video shares", size=18, weight="bold")
plt.setp(autotexts, size=14, weight="bold")

US_GB_data = [usGbCount, usCount-usGbCount, gbCount-usGbCount]
ingredients1 = ["COMMON", "USA ONLY", "GRATE BRITAIN ONLY"]

wedges, texts, autotexts1 = ax[0][1].pie(US_GB_data, autopct=lambda pct: func(pct, US_GB_data), textprops=dict(color="w"))
ax[0][1].legend(wedges, ingredients1, title="COUNTRY", loc="center", bbox_to_anchor=(1, 0.8, 0, 0))
ax[0][1].set_title("USA & Great Britain Trending Video shares", size=18, weight="bold")
plt.setp(autotexts1, size=14, weight="bold")

US_IN_data = [usInCount, usCount-usInCount, inCount-usInCount]
ingredients2 = ["COMMON", "USA ONLY", "INDIA ONLY"]

wedges, texts, autotexts2 = ax[1][0].pie(US_IN_data, autopct=lambda pct: func(pct, US_IN_data), textprops=dict(color="w"))
ax[1][0].legend(wedges, ingredients2, title="COUNTRY", loc="center", bbox_to_anchor=(1, 0.8, 0, 0))
ax[1][0].set_title("USA & India Trending Video shares", size=18, weight="bold")
plt.setp(autotexts2, size=14, weight="bold")

GB_IN_data = [gbInCount, gbCount-gbInCount, inCount-gbInCount]
ingredients3 = ["COMMON", "GRATE BRITAIN ONLY", "INDIA ONLY"]

wedges, texts, autotexts3 = ax[1][1].pie(GB_IN_data, autopct=lambda pct: func(pct, GB_IN_data), textprops=dict(color="w"))
ax[1][1].legend(wedges, ingredients3, title="COUNTRY", loc="center", bbox_to_anchor=(1, 0.8, 0, 0))
ax[1][1].set_title("Great Britain & India Trending Video shares", size=18, weight="bold")
plt.setp(autotexts3, size=14, weight="bold")

plt.show()
fig.savefig("youTube_Trend.png",dpi=400, bbox_inches='tight')
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/3a3eebb4b7134b18ef4b5d401e94200623c7fb14.png)
:::

::: {.output .stream .stdout}
    time: 30.4 s (started: 2021-12-10 17:37:17 +00:00)
:::
:::

::: {.cell .markdown id="WrGzS2Qj48cB"}
##### e) Comparison between number of trending videos per category across 3 countries. {#e-comparison-between-number-of-trending-videos-per-category-across-3-countries}

1\) Count trending video for each category for 3 country

2\) Prepare Histogram for category wise country video Count for
comparision
:::

::: {.cell .code id="7enswACIO8fp"}
``` {.python}
```
:::

::: {.cell .code colab="{\"height\":721,\"base_uri\":\"https://localhost:8080/\"}" id="gQZvdsjdEEDg" outputId="63a29baf-13da-4b8a-e406-5b515f17e52a"}
``` {.python}
# Histogram for X: Number of likes (Vs) Y: number of Videos per bin/bucket.

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

usDropDupDF = uscdf.dropDuplicates(["video_id"])
usCategoryDF = usDropDupDF.join(usjdf, usDropDupDF.categoryId == usjdf.id, "left").select("video_id", upper(col('category')))\
               .withColumnRenamed("upper(category)", "category").na.replace('[NONE]', None).select("video_id", regexp_replace(col("category"), " ", ""))\
               .withColumnRenamed("regexp_replace(category,  , )", "category").groupBy("category").agg(count("category").alias("count")).orderBy("count", ascending=False)

VDFBAR1_1D = usCategoryDF.select("Category")
VDFBAR1_1DOUT = [data[0] for data in VDFBAR1_1D.select('Category').collect()]
#print(VDFBAR1_1BOUT)

VDFBAR2_1D = usCategoryDF.select("count")
VDFBAR2_1DOUT = [data[0] for data in VDFBAR2_1D.select('count').collect()]
#print(VDFBAR2_1BOUT)

plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=VDFBAR1_1DOUT,x=VDFBAR2_1DOUT,orient="h")
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Trending videos category wise - US")
```

::: {.output .execute_result execution_count="82"}
    Text(0.5, 1.0, 'Trending videos category wise - US')
:::

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/761fb2c824c261c2b8a4c74d82241feda473eb98.png)
:::

::: {.output .stream .stdout}
    time: 12.3 s (started: 2021-12-10 17:37:47 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":717,\"base_uri\":\"https://localhost:8080/\"}" id="XZz93eLmGqta" outputId="7eb1958c-7459-4dbf-8344-29f3a09d5fcb"}
``` {.python}
# Histogram for X: Number of likes (Vs) Y: number of Videos per bin/bucket.

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

gbDropDupDF = gbcdf.dropDuplicates(["video_id"])
gbCategoryDF = gbDropDupDF.join(gbjdf, gbDropDupDF.categoryId == gbjdf.id, "left").select("video_id", upper(col('category')))\
               .withColumnRenamed("upper(category)", "category").na.replace('[NONE]', None).select("video_id", regexp_replace(col("category"), " ", ""))\
               .withColumnRenamed("regexp_replace(category,  , )", "category").groupBy("category").agg(count("category").alias("count")).orderBy("count", ascending=False)

VDFBAR1_1D = gbCategoryDF.select("Category")
VDFBAR1_1DOUT = [data[0] for data in VDFBAR1_1D.select('Category').collect()]
#print(VDFBAR1_1BOUT)

VDFBAR2_1D = gbCategoryDF.select("count")
VDFBAR2_1DOUT = [data[0] for data in VDFBAR2_1D.select('count').collect()]
#print(VDFBAR2_1BOUT)

plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=VDFBAR1_1DOUT,x=VDFBAR2_1DOUT,orient="h")
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Trending videos category wise - Great Britain")
```

::: {.output .execute_result execution_count="83"}
    Text(0.5, 1.0, 'Trending videos category wise - Great Britain')
:::

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/76725c2397019ab6147e4e9a65940e47b0a19051.png)
:::

::: {.output .stream .stdout}
    time: 11.7 s (started: 2021-12-10 17:38:00 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":721,\"base_uri\":\"https://localhost:8080/\"}" id="WdRhgmeHLm81" outputId="e65c90c0-978c-4696-b3c4-59dfe7d16277"}
``` {.python}
# Histogram for X: Number of likes (Vs) Y: number of Videos per bin/bucket.

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

inDropDupDF = incdf.dropDuplicates(["video_id"])
inCategoryDF = inDropDupDF.join(injdf, inDropDupDF.categoryId == injdf.id, "left").select("video_id", upper(col('category')))\
               .withColumnRenamed("upper(category)", "category").na.replace('[NONE]', None).select("video_id", regexp_replace(col("category"), " ", ""))\
               .withColumnRenamed("regexp_replace(category,  , )", "category").groupBy("category").agg(count("category").alias("count")).orderBy("count", ascending=False)

VDFBAR1_1D = inCategoryDF.select("Category")
VDFBAR1_1DOUT = [data[0] for data in VDFBAR1_1D.select('Category').collect()]
#print(VDFBAR1_1BOUT)

VDFBAR2_1D = inCategoryDF.select("count")
VDFBAR2_1DOUT = [data[0] for data in VDFBAR2_1D.select('count').collect()]
#print(VDFBAR2_1BOUT)

plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=VDFBAR1_1DOUT,x=VDFBAR2_1DOUT,orient="h")
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Trending videos category wise - India")
```

::: {.output .execute_result execution_count="84"}
    Text(0.5, 1.0, 'Trending videos category wise - India')
:::

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/a198d545751f19ceb93f9aac1109d14349ff2f80.png)
:::

::: {.output .stream .stdout}
    time: 11.8 s (started: 2021-12-10 17:38:11 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="5CMSVBAH4NjL" outputId="b2fbd62e-09ee-4419-b042-e7dad7b27280"}
``` {.python}
# Section 4  - Starts Here
```

::: {.output .stream .stdout}
    time: 706 µs (started: 2021-12-10 17:38:23 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="y6GJpfDF4anh" outputId="93321359-8e3e-4e4b-e47a-6c9df9413fe2"}
``` {.python}
DF1 = usadf.select(split(usadf.tags, '[|]').alias("Tag_List"), col("tags"), col("video_id"),col("title"))

#VDF2.show(10)

##Make NULL as 0

DF2 = DF1.withColumn("NumberofTags", size(col("Tag_List")))
DF3 = DF2.withColumn("NumberofValidTags", when(DF2["NumberofTags"] == -1, 0).otherwise(DF2["NumberofTags"]))
DF4 = DF3.select("video_id", "title", "tags", "NumberofValidTags")
DF5 = DF4.agg(avg(col("NumberofValidTags")))
DF4.show(truncate=False)
#DF5.show()

print("Average Number of Tags")
DFHISTTAGS = DF4.select("NumberofValidTags")
DFHISTTAGSOUT = [data[0] for data in DFHISTTAGS.select('NumberofValidTags').collect()]
#print(VDFHIST1OUT)
```

::: {.output .stream .stdout}
    +-----------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------+
    |video_id   |title                                                                   |tags                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |NumberofValidTags|
    +-----------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------+
    |3C66w5Z0ixs|I ASKED HER TO BE MY GIRLFRIEND...                                      |brawadis|prank|basketball|skits|ghost|funny videos|vlog|vlogging|NBA|browadis|challenges|bmw i8|faze rug|faze rug brother|mama rug and papa rug                                                                                                                                                                                                                                                                                                                                     |15               |
    |M9Pmf9AB4Mo|Apex Legends | Stories from the Outlands – “The Endorsement”            |Apex Legends|Apex Legends characters|new Apex Legend|Apex Legends Rampart|Apex Legends Season 6|Apex Legends Boosted|Battle Pass|Season 6 Battle Pass|Apex Legends new season|Apex Legends game|Respawn Apex Legends|Battle Royale game|Battle Royale|Battle Royale shooter|Apex Games|squad play|multiplayer shooter|Apex Legends PS4|Apex Legends Xbox|Apex Legends PC|Apex Legends Origin|Respawn Entertainment|Electronic Arts|Titanfall 2|fun battle royale                    |25               |
    |J78aPJ3VyNs|I left youtube for a month and THIS is what happened.                   |jacksepticeye|funny|funny meme|memes|jacksepticeye memes|reddit|subreddit|community|community memes|community subreddit|jacksepticeye subreddit|reddit memes|fan submitted|spicy memes|funny pics|reaction|react|green screen|funny memes|funny green screen|dank memes|memes compilation|try not to laugh|meme|fresh memes|meme review|funny moments|bell memes|bell of meme|jacksepticeye bell                                                                                    |30               |
    |kXLn3HkpjaA|XXL 2020 Freshman Class Revealed - Official Announcement                |xxl freshman|xxl freshmen|2020 xxl freshman|2020 freshman|2020 freshmen|xxl freshman class|2020 xxl freshman class|NLE Choppa|Polo G|Chika|Baby Keem|Mulatto|Jack Harlow|Rod Wave|Lil Tjay|Calboy|Fivio Foreign|Lil Keed|24kGoldn|rapper|rap|hip-hop|music                                                                                                                                                                                                                          |23               |
    |VIUo6yapDbc|Ultimate DIY Home Movie Theater for The LaBrant Family!                 |The LaBrant Family|DIY|Interior Design|Makeover|Decorating|DIY Movie Theater|Home Movie Theater|Movie Theater|Stay Home|Cole LaBrant|Savannah LaBrant|Kate Albrecht|Joey Zehr|mr. Kate|mr kate|luxurious|red|beautiful|LifeLock Norton 360|Norton|LifeLock|Real estate scams|Home title scams|Identity Theft|Cyber Security|Cyber Safety|lux|collab|omg we're coming over|youtubers|home makeover|nursery makeover|home theater                                                     |33               |
    |w-aidBdvZo8|I Haven't Been Honest About My Injury.. Here's THE TRUTH                |Professor injury|professor achilles|professor 1v1|professor live|professor live injury|professorlive injury                                                                                                                                                                                                                                                                                                                                                                         |6                |
    |uet14uf9NsE|OUR FIRST FAMILY INTRO!!                                                |[None]                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |1                |
    |ua4QMFQATco|CGP Grey was WRONG                                                      |cgpgrey|education|hello internet                                                                                                                                                                                                                                                                                                                                                                                                                                                    |3                |
    |SnsPZj91R7E|SURPRISING MY DAD WITH HIS DREAM TRUCK!! | Louie's Life                 |surprising|dad|father|papa|with|dream|car|truck|troca|camioneta|surprise|surprising my dad|parents|mom|mexican|latino|latinos|spanish|espanol|dream truck|dream car|goals|accomplish|car tour|tour|chevrolet|2020|new|new chevrolet|interior|vlog|money|price|louie|castro|louie castro|louies life|the baddest perra|spanglish|louies family|castro sisters|mua|makeup|surprising my mom                                                                                           |44               |
    |SsWHMAhshPQ|Ovi x Natanael Cano x Aleman x Big Soto - Vengo De Nada [Official Video]|Vengo De Nada|Aleman|Ovi|Big Soto|Trap|Ovi Natanel Cano Big Soto Aleman|Nata Aleman|Ovi Vengo De Nada|Natanael Cano Vengo De Nada|Aleman Vengo De Nada|Big Soto Vengo De Nada|Natanael Cano|Nata|Legado 7|Herencia de Patrones|El De La Guitarra|Grupo Codiciado|Alta Consigna|Junior H|Cosas de la Clica|Pacas Verdes|El Drip|Corridos Tumbados|Jimmy Humilde|Rancho Humilde|Rancho|Corridos|Corridos Verdes|Regional urbano|Clika|SMO|Regional Mexicano|Banda                     |33               |
    |49Z6Mv4_WCA|i don't know what im doing anymore                                      |[None]                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |1                |
    |nt3VVyv5pxQ|Try Not To Laugh Challenge #51                                          |smosh|smosh pit|smosh games|funny|comedy                                                                                                                                                                                                                                                                                                                                                                                                                                            |5                |
    |I6hswz4rIrU|Rainbow Six Siege: Operation Shadow Legacy Reveal Trailer | Ubisoft [NA]|R6|R6S|Siege|New Siege|New Operators|New Ops|Gameplay|Rainbow Six New Operators|R6S New Operation|Ubisoft|Ubi|Operation Shadow Legacy|Shadow Legacy|R6S Shadow Legacy|Rainbow 6 Shadow Legacy|Year 5 Season 3|Y5S3|Sam Fisher|Splinter Cell|Rainbow 6 Sam Fisher|Zero|R6 Zero|R6S Zero|Rainbow 6 Zero                                                                                                                                                                               |24               |
    |W7VK4DUHvKU|Lil Yachty & Future - Pardon Me (Official Video)                        |Lil Yachty|Lil Boat 3|Future Lil Yachty|Pardon Me Future|Pardon Me Lil Yachty|Lil Yachty future pardon me                                                                                                                                                                                                                                                                                                                                                                           |6                |
    |W9Aen8hG20Y|When Our Generation Gets Old and Hears a Throwback Song 5               |When Our Generation Gets Old and Hears a Throwback Song 5|When Our Generation Gets Old and Hears a Throwback Song|When Our Generation Gets Old and Hears a Throwback Song 3|When Our Generation Gets Old and Hears a Throwback Song 4|Kyle Exum|Exum|Bassthoven|Generation Old|Kyle TikTok|trap 3 little pigs|kyle exum parody songs|starbucks be like|gooba|6ix9ine|blueberry faygo|lil mosey|apple be like|songs|old belt road|mama mode|parents be like|exumseason|tiktokin|kyle |24               |
    |BNeDH6UTmXw|Ten Minutes with Tyler Cameron | Q&A                                    |the bachelor|the bachelorette|Tyler c|Tyler Cameron|Tyler c bachelorette|Tyler c bachelor|Tyler Cameron bachelorette|Tyler Cameron bachelor|Hannah brown|Hannah brown bachelorette|Tyler c and Hannah|Tyler c and Hannah b|Tyler c and Hannah bachelorette|Tyler Cameron and Hannah brown|Tyler Cameron and Hannah brown bachelorette|pop culture|Gigi Hadid|Tyler c and Gigi Hadid                                                                                                 |18               |
    |6TIsR_7nrNc|Kylie Jenner Reacts To 'WAP' Music Video Backlash                       |kylie jenner|kendall jenner|cardi b|wap|reacts|video|kuwtk|skims|tiktok                                                                                                                                                                                                                                                                                                                                                                                                             |9                |
    |gPdUslndvVI|Our Farm Got Destroyed.                                                 |farming|family farm|agriculture|agriculture jobs|mechanic|welding|fabricating|tractors|tractor videos|renovation|hoarders|scrappers|junkyard|old farm house|vacant property|abandoned house|remodeling|house renovation|satisfying videos|funny videos|diy|farmstead|homestead|cleaning|big equipment|heavy machinery|trucks|combines|harvest|stuck equipment|dirty jobs|John Deere|Case IH|house projects|projects|best vlogs|renovating|midwest|semis|hard work|Jesus|Faith       |42               |
    |GTp-0S82guE|Time to Talk..                                                          |chloe ting|chloeting|defamation|cyber bullying|vlog|chloetingchallenge|online bullying|chloe ting expose                                                                                                                                                                                                                                                                                                                                                                            |8                |
    |jbGRowa5tIk|ITZY “Not Shy” M/V TEASER                                               |JYP Entertainment|JYP|ITZY|있지|ITZY Video|ITZY Yeji|ITZY Lia|ITZY Ryujin|ITZY Chaeryeong|ITZY Yuna|있지 예지|있지 리아|있지 류진|있지 채령|있지 유나|예지|리아|류진|채령|유나|Yeji|Lia|Ryujin|Chaeryeong|Yuna|ITZY BEHIND|ITZY VIDEOS|ITZY DEBUT|있지 데뷔|not shy|ITZY not shy|ITZY TEASER|있지 티저|not shy 티저|있지 뮤비 티저|not shy 뮤비 티저|ITZY MV TEASER|ITZY MV|M/V|TEASER|있지 컴백|ITZY COMEBACK|not shy MV|낫샤이 뮤비|낫샤이|not shy opening trailer|opening trailer|47               |
    +-----------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------+
    only showing top 20 rows

    Average Number of Tags
    time: 3.17 s (started: 2021-12-10 17:38:23 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":527,\"base_uri\":\"https://localhost:8080/\"}" id="mcknz_DUm2W6" outputId="1cf5abe8-a7a6-4c5e-9976-cf5bed7797be"}
``` {.python}
#Histogram Tags Length
from matplotlib import pyplot as plt
import numpy as np
fig,ax = plt.subplots(1,1)
ax.hist(DFHISTTAGSOUT, bins = [0,10,20,30,40,50,60,70,80,90,100])
ax.set_title("Histogram of Number of Tags")
ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
ax.set_xlabel('Number of Tags')
ax.set_ylabel('Number of Videos')
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/d991b84e29eb59abbe46672b4ead8b0330c60892.png)
:::

::: {.output .stream .stdout}
    time: 293 ms (started: 2021-12-10 17:38:26 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="7m4ce2BKm76L" outputId="0e1a53c7-e695-43a0-8d2d-794705f42221"}
``` {.python}
#Average Title Number of Words

DF102 = usadf.select(split(usadf.title, ' ').alias("title_List"), col("video_id"),col("title"))
#DF101.show(10)

##Make NULL as 0
DF103 = DF102.withColumn("NumberofWords", size(col("title_List")))
DF104 = DF103.withColumn("NumberofValidWords", when(DF103["NumberofWords"] == -1, 0).otherwise(DF103["NumberofWords"]))
DF105 = DF104.select("video_id", "title", "NumberofValidWords").agg(avg(col("NumberofValidWords")))
#VDF104.show(10)
DF105.show(10)

print("Average Number of Words in Title")
DFHISTTITLEWORDS = DF104.select("NumberofValidWords")
DFHISTTITLEWORDSOUT = [data[0] for data in DFHISTTITLEWORDS.select('NumberofValidWords').collect()]
#print(DFHISTTITLEWORDSOUT)
```

::: {.output .stream .stdout}
    +-----------------------+
    |avg(NumberofValidWords)|
    +-----------------------+
    |      8.853536839123661|
    +-----------------------+

    Average Number of Words in Title
    time: 3.94 s (started: 2021-12-10 17:38:27 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":527,\"base_uri\":\"https://localhost:8080/\"}" id="imMU8MeDnCd3" outputId="f244edad-d7c0-43b5-dffe-79c51947c606"}
``` {.python}
#When NULL Values are made 0
from matplotlib import pyplot as plt
import numpy as np
fig,ax = plt.subplots(1,1)
ax.hist(DFHISTTITLEWORDSOUT, bins = [0,5,10,15,20,25,30])
ax.set_title("Histogram of Number of Words in Title")
ax.set_xticks([0,5,10,15,20,25,30])
ax.set_xlabel('Number of Words in Title')
ax.set_ylabel('Number of Videos')
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/5dab54a7ab1b56d8afaaea065f753e43472a52ef.png)
:::

::: {.output .stream .stdout}
    time: 243 ms (started: 2021-12-10 17:38:31 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="mjXVWxagnGl2" outputId="5e0db933-d4c7-4366-8f2c-1c9e70140c0b"}
``` {.python}
#TOP 50 Tags (sentences) in General

VDF202 = usadf.filter(col("tags").isNotNull()).select(split(usadf.tags, '[|]').alias("tags_list"), col("video_id"),col("title"))
VDF203 = VDF202.select("video_id", "title", explode(VDF202.tags_list).alias("tags")).filter(col("tags") != "[None]")
VDF204 = VDF203.groupby("tags").agg(count("tags")).withColumnRenamed("tags", "TOP 50 Tags").withColumnRenamed("count(tags)", "Number of Apperances").sort(col("Number of Apperances").desc())
VDF205 = VDF204.select("TOP 50 Tags", "Number of Apperances").limit(50)
VDF205.show(50)
```

::: {.output .stream .stdout}
    +-------------------+--------------------+
    |        TOP 50 Tags|Number of Apperances|
    +-------------------+--------------------+
    |              funny|                5830|
    |             comedy|                3781|
    |          minecraft|                3098|
    |          challenge|                2349|
    |               vlog|                2022|
    |               news|                1827|
    |          animation|                1726|
    |                rap|                1700|
    |              music|                1608|
    |             gaming|                1574|
    |         highlights|                1566|
    |             tiktok|                1531|
    |            hip hop|                1425|
    |           reaction|                1407|
    |             how to|                1396|
    |             family|                1352|
    |               2020|                1337|
    |           football|                1333|
    |           fortnite|                1301|
    |                new|                1253|
    |               game|                1205|
    |              video|                1200|
    |    family friendly|                1187|
    |             sports|                1154|
    |                Rap|                1121|
    |                NBA|                1087|
    |                fun|                1073|
    |            tik tok|                1017|
    |           Football|                 995|
    |               2021|                 974|
    |          Minecraft|                 972|
    |             review|                 930|
    |            Records|                 918|
    |             soccer|                 915|
    |            Hip Hop|                 909|
    |        music video|                 905|
    |            trailer|                 859|
    |            science|                 848|
    |minecraft challenge|                 840|
    |           among us|                 839|
    |              dream|                 832|
    |                nba|                 803|
    |          interview|                 797|
    |           gameplay|                 795|
    |              Music|                 794|
    |               live|                 782|
    |             parody|                 777|
    |                diy|                 767|
    |               play|                 742|
    |           reacting|                 729|
    +-------------------+--------------------+

    time: 4.02 s (started: 2021-12-10 17:38:31 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="NoGbypxFnQaH" outputId="61146e14-548f-410e-ca8f-557a9a497c57"}
``` {.python}
#TOP 50 Tags (sentences) > 1M Views 

VDF302 = usadf.filter(col("tags").isNotNull() & col("view_count").isNotNull()).select(split(usadf.tags, '[|]').alias("tags_list"), col("video_id"),col("title"), col("view_count")).filter(col("view_count") > 1000000)
VDF303 = VDF302.select("video_id", "title", explode(VDF302.tags_list).alias("tags")).filter(col("tags") != "[None]")
VDF304 = VDF303.groupby("tags").agg(count("tags")).withColumnRenamed("tags", "TOP 50 Tags with More than 1M Views").withColumnRenamed("count(tags)", "Number of Apperances").sort(col("Number of Apperances").desc())
VDF305 = VDF304.select("TOP 50 Tags with More than 1M Views", "Number of Apperances").limit(50)
VDF305.show(50)
```

::: {.output .stream .stdout}
    +-----------------------------------+--------------------+
    |TOP 50 Tags with More than 1M Views|Number of Apperances|
    +-----------------------------------+--------------------+
    |                              funny|                3460|
    |                          minecraft|                2043|
    |                             comedy|                1977|
    |                          challenge|                1547|
    |                         highlights|                1013|
    |                             tiktok|                1007|
    |                               vlog|                1006|
    |                           reaction|                 988|
    |                             family|                 853|
    |                          animation|                 848|
    |                           football|                 839|
    |                               game|                 826|
    |                               2020|                 821|
    |                    family friendly|                 807|
    |                                fun|                 802|
    |                                rap|                 795|
    |                             gaming|                 775|
    |                              music|                 758|
    |                            tik tok|                 729|
    |                                new|                 720|
    |                                NBA|                 705|
    |                           fortnite|                 690|
    |                          Minecraft|                 679|
    |                              video|                 676|
    |                minecraft challenge|                 668|
    |                            hip hop|                 666|
    |                               news|                 639|
    |                                BTS|                 639|
    |                                Pop|                 636|
    |                           reacting|                 634|
    |                                Rap|                 627|
    |                            Records|                 623|
    |                               play|                 621|
    |                           among us|                 620|
    |                           Football|                 620|
    |                              dream|                 612|
    |                        music video|                 609|
    |                             how to|                 605|
    |                         Basketball|                 590|
    |                             soccer|                 569|
    |                           G League|                 560|
    |                            Hip Hop|                 557|
    |                             matpat|                 547|
    |                            science|                 532|
    |                              laugh|                 522|
    |                               live|                 507|
    |                          highlight|                 500|
    |                       sssniperwolf|                 498|
    |                             parody|                 495|
    |                         Highlights|                 494|
    +-----------------------------------+--------------------+

    time: 3.17 s (started: 2021-12-10 17:38:35 +00:00)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="cShdC_yvnexn" outputId="40ce9b49-07e6-4c83-cae3-e3f34e8a65f7"}
``` {.python}
#TOP 50 Words in Tags in Videos > 1M Views 

VDF402 = usadf.filter(col("tags").isNotNull() & col("view_count").isNotNull()).select(split(usadf.tags, '[| ]').alias("tags_list"), col("video_id"),col("title"), col("view_count")).filter(col("view_count") > 1000000)
VDF403 = VDF402.select("video_id", "title", explode(VDF402.tags_list).alias("tags")).filter(col("tags") != "[None]")
VDF404 = VDF403.groupby("tags").agg(count("tags")).withColumnRenamed("tags", "TOP 50 Tags Words with More than 1M Views").withColumnRenamed("count(tags)", "Number of Apperances").sort(col("Number of Apperances").desc())
VDF405 = VDF404.select("TOP 50 Tags Words with More than 1M Views", "Number of Apperances").limit(50)
VDF405.show(50)
```

::: {.output .stream .stdout}
    +-----------------------------------------+--------------------+
    |TOP 50 Tags Words with More than 1M Views|Number of Apperances|
    +-----------------------------------------+--------------------+
    |                                minecraft|               16100|
    |                                      the|               13968|
    |                                    video|                9753|
    |                                      new|                8567|
    |                                     game|                8435|
    |                                       us|                7668|
    |                                    funny|                7452|
    |                                      and|                6998|
    |                               highlights|                6756|
    |                                    among|                6386|
    |                                    music|                6234|
    |                                       to|                6190|
    |                                       of|                6155|
    |                                       in|                6009|
    |                                     2021|                5686|
    |                                     2020|                5608|
    |                                Minecraft|                5605|
    |                                       vs|                5588|
    |                                        a|                5071|
    |                                 fortnite|                4720|
    |                                challenge|                4124|
    |                                  trailer|                4112|
    |                                      The|                3974|
    |                                   season|                3693|
    |                                 official|                3502|
    |                                   family|                3458|
    |                                   tiktok|                3336|
    |                                    night|                3281|
    |                                   makeup|                3244|
    |                                        2|                3242|
    |                                     live|                3220|
    |                                       me|                3168|
    |                                     news|                3160|
    |                                     life|                3138|
    |                                      how|                3109|
    |                                     Apex|                3069|
    |                                      you|                2958|
    |                                      but|                2920|
    |                                      lil|                2855|
    |                                  Legends|                2834|
    |                                      100|                2828|
    |                                     best|                2808|
    |                                     paul|                2771|
    |                                     song|                2764|
    |                                animation|                2736|
    |                                       on|                2641|
    |                                       my|                2575|
    |                                   comedy|                2574|
    |                                    Music|                2567|
    |                                   theory|                2525|
    +-----------------------------------------+--------------------+

    time: 3.57 s (started: 2021-12-10 17:38:38 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":588,\"base_uri\":\"https://localhost:8080/\"}" id="5PjESf_knrgM" outputId="8ea04f9d-ec78-4803-b02a-79f5bc15ad08"}
``` {.python}
#WordCloud

VDF502 = usadf.filter(col("tags").isNotNull() & col("view_count").isNotNull()).filter(col("tags") != "[None]").select(split(usadf.tags, '[| ]').alias("tags_list"), col("video_id"),col("title"), col("view_count")).filter(col("view_count") > 1000000)
#VDF502.show()
VDFWORDOUT = [data[0] for data in VDF502.select('tags_list').collect()]
VDF503 = VDF502.select("video_id", "title", explode(VDF502.tags_list).alias("tags")).filter(col("tags") != "[None]")
VDF504 = VDF503.groupby("tags").agg(count("tags")).withColumnRenamed("tags", "TOP 50 Tags Words with More than 1M Views").withColumnRenamed("count(tags)", "Number of Apperances").sort(col("Number of Apperances").desc())
VDF505 = VDF504.select("TOP 50 Tags Words with More than 1M Views", "Number of Apperances").limit(50)
#VDF505.show(100)
#print(VDFWORDOUT)
tags_words = ''
for val in VDFWORDOUT:
  tags_words += " ".join(val)+" " 

#print(tags_words)
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 1500, height = 1000,
                background_color ='white',
                stopwords = stopwords,collocations = False,
                min_font_size = 10).generate(tags_words)
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()
```

::: {.output .display_data}
![](vertopal_29957940c80e431bbbbb22b85783604c/1d5b852259c3298c059b8c535b75548e9ddbbb93.png)
:::

::: {.output .stream .stdout}
    time: 11.8 s (started: 2021-12-10 17:38:42 +00:00)
:::
:::

::: {.cell .markdown id="Ejs_jjDZn-6A"}
##### Video Recommendation
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="mtw2N5eJn2hm" outputId="73dae85e-d666-4de0-bbd6-8ed35f9cc758"}
``` {.python}
#Check Duplicate Video ID

print("Total Number of ROWS", usadf.count())

#print("Duplicate Video ID List")

#df1=VDF.groupBy("video_id").count().filter("count > 1")
#df1.drop('count').sort("video_id").show(400)


#print("Duplicate Video ID Count")

#df1=VDF.groupBy("video_id").count().filter("count > 1")
#df1.sort("video_id").show(400)


print("Total Number of ROWS After removing duplicates", usadf.dropDuplicates(["video_id"]).dropDuplicates(["title"]).count())

print("Category of Videos in DataSet")

ytvusjdf.show(100)

#Drop Duplicate Video ID and Titles

VDFdropDupDF = usadf.dropDuplicates(["video_id"]).dropDuplicates(["title"])

#Get Tags ordered

df1000 = VDFdropDupDF.join(ytvusjdf, VDFdropDupDF.categoryId == ytvusjdf.id, "left").select("video_id", upper(col('category')), upper(col('tags')), "title", "likes", "dislikes").withColumnRenamed("upper(tags)", "tags").withColumnRenamed("upper(category)", "category").na.replace('[NONE]', None)
df1001_0 = df1000.select("video_id", "title", "likes", "dislikes", regexp_replace(col("category"), " ", ""), regexp_replace(col("tags"), " ", "")).withColumnRenamed("regexp_replace(category,  , )", "category").withColumnRenamed("regexp_replace(tags,  , )", "tags")
df1001 = df1001_0.select("video_id", "category", "title", "likes", "dislikes", split(df1001_0.tags, '[|]', limit=4).alias("tags_list"))
df1002 = df1001.select("video_id", "category", "title", "tags_list", "likes", "dislikes", split(df1001.category, '[|]').alias("category_list"))
df1003 = df1002.select("video_id","category_list","title","tags_list", "likes", "dislikes", concat(df1002.category_list,df1002.tags_list).alias("combined"))
df1004 = df1003.withColumn("combined",coalesce(df1003.combined,df1003.category_list)) 
df1005 = df1004.withColumn("Len", size("combined")).withColumn("Percentage Likes", col("likes")/col("dislikes"))
#df1005.show(truncate=False)
#df1003.show(truncate=False)


#Getting Only Videos Which are related

#df2000 = df1005.select("title", "combined").withColumn("combined_small", array([col("combined")[0], col("combined")[1], col("combined")[2], col("combined")[3]])).limit(10)
df2000 = df1005.select("title", array([col("combined")[0], col("combined")[1], col("combined")[2], col("combined")[3]]).alias("combined_small")).limit(1000)
df2001 = df2000.withColumn("ID", monotonically_increasing_id())
#df2001.show(20,truncate= False)

from pyspark.ml.feature import HashingTF, IDF
hashingTF = HashingTF(inputCol="combined_small", outputCol="tf")
#hashingTF = HashingTF(inputCol="combined", outputCol="tf")
tf = hashingTF.transform(df2001)

idf = IDF(inputCol="tf", outputCol="feature").fit(tf)
tfidf = idf.transform(tf)

#Compute L2 Norm

from pyspark.ml.feature import Normalizer
normalizer = Normalizer(inputCol="feature", outputCol="norm")
data = normalizer.transform(tfidf)

import pyspark.sql.functions as psf
from pyspark.sql.types import DoubleType
dot_udf = psf.udf(lambda x,y: float(x.dot(y)), DoubleType())
final = data.alias("i").join(data.alias("j"), psf.col("i.ID") < psf.col("j.ID"))\
    .select(psf.col("i.ID").alias("i"), psf.col("j.ID").alias("j"), dot_udf("i.norm", "j.norm").alias("dot"))\
    .where(col('dot') != 0).sort("i", "j")
#final.show()


final_video = final.select("j", "i", "dot")
final_video2 = final.union(final_video)
final_video3 = final_video2.sort("i", "dot", ascending=False)
#final3.show()


pandasDF = df2001.toPandas()
sparkDF=spark.createDataFrame(pandasDF)


final_video4 = final_video3.join(sparkDF, sparkDF.ID == final_video3.i).withColumnRenamed("title", "Search Title").drop("ID","combined_small")
final_video5 = final_video4.join(sparkDF, sparkDF.ID == final_video4.j).withColumnRenamed("title", "Recommendation Title").drop("ID")
#final_video4.show(100, truncate=False)
#final_video5.show(100)

pandasDFRecommendation = final_video5.toPandas()
#pandasDFRecommendation

```

::: {.output .stream .stdout}
    Total Number of ROWS 91791
    Total Number of ROWS After removing duplicates 16009
    Category of Videos in DataSet
    +---+--------------------+
    | id|            Category|
    +---+--------------------+
    |  1|    Film & Animation|
    |  2|    Autos & Vehicles|
    | 10|               Music|
    | 15|      Pets & Animals|
    | 17|              Sports|
    | 18|        Short Movies|
    | 19|     Travel & Events|
    | 20|              Gaming|
    | 21|       Videoblogging|
    | 22|      People & Blogs|
    | 23|              Comedy|
    | 24|       Entertainment|
    | 25|     News & Politics|
    | 26|       Howto & Style|
    | 27|           Education|
    | 28|Science & Technology|
    | 29|Nonprofits & Acti...|
    | 30|              Movies|
    | 31|     Anime/Animation|
    | 32|    Action/Adventure|
    | 33|            Classics|
    | 34|              Comedy|
    | 35|         Documentary|
    | 36|               Drama|
    | 37|              Family|
    | 38|             Foreign|
    | 39|              Horror|
    | 40|      Sci-Fi/Fantasy|
    | 41|            Thriller|
    | 42|              Shorts|
    | 43|               Shows|
    | 44|            Trailers|
    +---+--------------------+

    time: 5min 37s (started: 2021-12-10 17:38:54 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="u9wRNvLliw5Z" outputId="9f054cec-a507-4e1a-9042-42dcbe31a1b2"}
``` {.python}
pandasDF.to_csv("videolist")
pandasDF
```

::: {.output .execute_result execution_count="108"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>combined_small</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>*NEW* FERRARI in FORTNITE!</td>
      <td>[GAMING, FORTNITE, FORTNITEBR, FORTNITEBATTLEROYALE]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10 Things Not To Do in SQUID GAME</td>
      <td>[ENTERTAINMENT, GUAVAJUICE, GUAVAJUICEYOUTUBE, YOUTUBEGUAVAJUICE]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41 And PREGNANT</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7 fairly uninteresting projects I've built</td>
      <td>[SCIENCE&amp;TECHNOLOGY, None, None, None]</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A VERY EXTRA WISH HAUL.. maybe the most extra ever ?!</td>
      <td>[HOWTO&amp;STYLE, MIAMAPLES, WISHHAUL, VERYEXTRAWISHCLOTHINGHAUL]</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Agent Orange (The Vietnam War)</td>
      <td>[EDUCATION, SIMPLEHISTORY, ANIMATEDHISTORY, EDUCATIONAL]</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Alemán - Mi Tio Snoop Ft Snoop Dogg (Video Oficial)</td>
      <td>[MUSIC, ALEMÁN, ALEMANHOMEGROWN, HOMEGROWNMAFIA]</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Andrea Bocelli launches EURO 2020!</td>
      <td>[SPORTS, THE_LATEST, UEFA, EURO]</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Argentina 3-0 Uruguay I Eliminatorias a Catar 2022</td>
      <td>[SPORTS, TYC, SPORTS, TYCSPORTS]</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BTS React To Their First Hot 100 No. 1 Hit ‘Dynamite’ and Tease What's Next | Billboard News</td>
      <td>[MUSIC, BILLBOARD, BILLBOARDCHANNEL, OFFICIAL]</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BTS | Meet The First-Time GRAMMY Nominees</td>
      <td>[MUSIC, GRAMMYAWARDS, GRAMMYS, GRAMMY]</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Browns vs. Chiefs Week 1 Highlights | NFL 2021</td>
      <td>[SPORTS, NFL, FOOTBALL, OFFENSE]</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Catching up with Da Rock’s Siblings... she a model now and I’m still a baddie</td>
      <td>[PEOPLE&amp;BLOGS, BRETMAN, BRETMANROCK, FUNNY]</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Celebrating Black Creativity with Guest Artist Marco Cheatham</td>
      <td>[EDUCATION, BLACK, BLACKHISTORY, CREATIVITY]</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Commercials From My Nightmares (flashing lights 10:40-10:50)</td>
      <td>[COMEDY, DANNYGONZALEZ, FUNNY, COMMENTARY]</td>
      <td>14</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DDG Moonwalking In Calabasas (Remix) Official Lyrics &amp; Meaning | Verified</td>
      <td>[MUSIC, GENIUS, RAPGENIUS, VERIFIED]</td>
      <td>15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Dana White sends HEATED message after UFC 266 comment, Conor McGregor trashes Alexander Volkanovski</td>
      <td>[SPORTS, MMA, UFC, MMANEWS]</td>
      <td>16</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Dax - KILLSHOT 3 (Official Music Video)</td>
      <td>[MUSIC, DAX, KILLSHOT, KILLSHOT3]</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Do NBA Superstars Get Preferential Treatment? | NBA on TNT</td>
      <td>[SPORTS, NBAONTNT, NBA, INSIDETHENBA]</td>
      <td>18</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Drake &amp; K Showtime Team Up &amp; GO CRAZY! 2v2 Basketball</td>
      <td>[MUSIC, None, None, None]</td>
      <td>19</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ENHYPEN Dance JAM Live #210717</td>
      <td>[PEOPLE&amp;BLOGS, BELIFTLAB, ENHYPEN, 엔하이픈]</td>
      <td>20</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ESCAPING 100 Layers of ICE vs CRAYONS!</td>
      <td>[ENTERTAINMENT, FUNNY, CHALLENGE, FAMILY]</td>
      <td>21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Early Release: Dustin Tavella Performs Heartwarming Magic - America's Got Talent 2021</td>
      <td>[ENTERTAINMENT, HILARIOUSMUSICSONGS, HIGHLIGHTS, SIMONCOWELL]</td>
      <td>22</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Exploring Why This Nuclear Fusion Breakthrough Matters</td>
      <td>[SCIENCE&amp;TECHNOLOGY, NUCLEARFUSION, NUCLEARFUSIONANDFISSION, NUCLEARFUSIONBREAKTHROUGH]</td>
      <td>23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>FC METZ - PARIS SAINT-GERMAIN (1 - 2) - Highlights - (FCM - PSG) / 2021-2022</td>
      <td>[SPORTS, FCMETZPARISSG, METZPARIS, HIGHLIGHTSFCMETZPARISSAINT-GERMAIN]</td>
      <td>24</td>
    </tr>
    <tr>
      <th>25</th>
      <td>GARCELLO Imposter Role in Among Us...</td>
      <td>[GAMING, AMONGUS, AMONGUSLOGIC, AMONGUSADVENTURES]</td>
      <td>25</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Game Theory: Friday Night Funkin Just BROKE Its Own Lore!</td>
      <td>[GAMING, FRIDAYNIGHTFUNKIN, FRIDAYNIGHTFUNKIN', HENRYSTICKMIN]</td>
      <td>26</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Gera MX, Christian Nodal - Botella Tras Botella (Video Oficial)</td>
      <td>[MUSIC, GERAMX, GERAMXM, None]</td>
      <td>27</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Get Glam With SELENA GOMEZ &amp; Me! | NikkieTutorials</td>
      <td>[HOWTO&amp;STYLE, SELENAGOMEZ, SELENAGOMEZMAKEUP, SELENA]</td>
      <td>28</td>
    </tr>
    <tr>
      <th>29</th>
      <td>HIDE &amp; SEEK ON KSI’s PRIVATE JET</td>
      <td>[PEOPLE&amp;BLOGS, SIDEMEN, MORESIDEMEN, MINIMINTER]</td>
      <td>29</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Happy Birthday, John</td>
      <td>[MUSIC, None, None, None]</td>
      <td>30</td>
    </tr>
    <tr>
      <th>31</th>
      <td>How Weird Is My Audience? I Polled 15,408 People To Find Out</td>
      <td>[EDUCATION, TOMSCOTT, TOMSCOTT, None]</td>
      <td>31</td>
    </tr>
    <tr>
      <th>32</th>
      <td>How to Replace a Front or Rear Wheel Bearing (Full ASMR)</td>
      <td>[AUTOS&amp;VEHICLES, CHRISFIX, CARASMR, ASMR]</td>
      <td>32</td>
    </tr>
    <tr>
      <th>33</th>
      <td>I Became the Mayor of Skyblock</td>
      <td>[GAMING, MINECRAFT, TECHNOBLADE, TECHNOTHEPIG]</td>
      <td>33</td>
    </tr>
    <tr>
      <th>34</th>
      <td>I Bought Every Seat On An Airplane!</td>
      <td>[ENTERTAINMENT, THE21000FIRSTCLASSAIRPLANESEAT, None, None]</td>
      <td>34</td>
    </tr>
    <tr>
      <th>35</th>
      <td>I Busted YOUR Minecraft Myths..</td>
      <td>[GAMING, MINECRAFTMYTHBUSTERS, MINECRAFTMYTHS, MINECRAFTMYTH]</td>
      <td>35</td>
    </tr>
    <tr>
      <th>36</th>
      <td>I Fixed My Masterpiece</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>36</td>
    </tr>
    <tr>
      <th>37</th>
      <td>I am so sorry..</td>
      <td>[PEOPLE&amp;BLOGS, NEW, UPLOAD, ROMANATWOOD]</td>
      <td>37</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Internet Money – JETSKI ft. Lil Mosey &amp; Lil Tecca (Official Video)</td>
      <td>[MUSIC, LILTECCA, LILMOSEY, LILUZIVERT]</td>
      <td>38</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Jackboy - Made it Out (Official Video)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>39</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Jaguars vs. Bengals Week 4 Highlights | NFL 2021</td>
      <td>[SPORTS, None, None, None]</td>
      <td>40</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Jamie Dornan Hilariously Rips Into Emily Blunt's Attempt At Pouring Guinness | Graham Norton Show</td>
      <td>[ENTERTAINMENT, GRAHAMNORTON, GRAHAMNORTONSHOW, THEGRAHAMNORTONSHOW]</td>
      <td>41</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Jim Carrey and Maya Rudolph Transform into Joe Biden and Kamala Harris - SNL</td>
      <td>[ENTERTAINMENT, SNL, SATURDAYNIGHTLIVE, SNL46]</td>
      <td>42</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Kay Flock - PSA (Official Video)</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>43</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Kendall Jenner Opens Up About Her Anxiety | Open Minded | Vogue</td>
      <td>[PEOPLE&amp;BLOGS, DRRAMANI, DRRAMANIDURVASULA, DRRAMANIDURVASULAANXIETY]</td>
      <td>44</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Kilauea Eruption Day Four - Rising Lava Lake Update (Dec. 24, 2020)</td>
      <td>[NEWS&amp;POLITICS, KILAUEA, VOLCANO, HAWAII]</td>
      <td>45</td>
    </tr>
    <tr>
      <th>46</th>
      <td>King Von - How It Go (Audio)</td>
      <td>[MUSIC, KINGVON, KINGVON2018, KINGVONOFFICIAL]</td>
      <td>46</td>
    </tr>
    <tr>
      <th>47</th>
      <td>LeBron James IMPRESSED By Andre Drummond &amp; Lakers Destroying Nets! Lakers vs Nets</td>
      <td>[SPORTS, NBA, CLIVENBAPARODY, SPORTS]</td>
      <td>47</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Luring My Friends To Their Death in Proximity Chat Among Us | CORPSE 666IQ IMPOSTER PLAYS</td>
      <td>[ENTERTAINMENT, AMONGUS, CORPSEAMONGUS, AMONGUSCORPSE]</td>
      <td>48</td>
    </tr>
    <tr>
      <th>49</th>
      <td>MAXING OUT 2020: How My Strength Keeps Skyrocketing</td>
      <td>[ENTERTAINMENT, JEFFNIPPARDAMRAP, JEFFNIPPARDMAXTESTING, JEFFNIPPARDMAXINGOUT]</td>
      <td>49</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MEMBRANE SWEEP LED ME INTO THE HOSPITAL...</td>
      <td>[PEOPLE&amp;BLOGS, MARIAESTELLA, USALWAYS, USALWAYSFAM]</td>
      <td>50</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Magnum - Nicky Jam x Jhay Cortez | Video Oficial</td>
      <td>[MUSIC, NICKYJAM, NICKYJAMPR, PIENSASENMI]</td>
      <td>51</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Maximum Jackman</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>52</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Merkules &amp; Jelly Roll - ''Twisted''</td>
      <td>[MUSIC, None, None, None]</td>
      <td>53</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Minecraft Live: Vote for the Iceologer!</td>
      <td>[GAMING, MINECRAFTLIVE, MINECRAFTLIVE2020, MOBVOTE]</td>
      <td>54</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Minecraft, But I Can Craft Any Item...</td>
      <td>[GAMING, MINECRAFT, MINECRAFTBUT, MCBUT]</td>
      <td>55</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Minecraft, But The World is Upside Down!</td>
      <td>[PEOPLE&amp;BLOGS, KARLJACOBS, None, None]</td>
      <td>56</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Monsters At Work | Official Trailer | Disney+</td>
      <td>[ENTERTAINMENT, DISNEY+, DISNEYPLUS, DISNEY]</td>
      <td>57</td>
    </tr>
    <tr>
      <th>58</th>
      <td>NASA Science Live: We Landed on Mars</td>
      <td>[SCIENCE&amp;TECHNOLOGY, None, None, None]</td>
      <td>58</td>
    </tr>
    <tr>
      <th>59</th>
      <td>NHL Highlights | Penguins @ Flyers 1/13/21</td>
      <td>[SPORTS, PHILADELPHIAFLYERS, PITTSBURGHPENGUINS, PHILADELPHIAFLYERSVS.PITTSBURGHPENGUINS]</td>
      <td>59</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Pelé 'Air Punch' Emote Coming to Fortnite</td>
      <td>[GAMING, FORTNITE, EPICGAMES, PC]</td>
      <td>60</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Pink Religion 💖 Palette &amp; Collection Reveal! | Jeffree Star Cosmetics</td>
      <td>[PEOPLE&amp;BLOGS, JEFFREESTAR, JEFFREESTARCOSMETICS, EYESHADOW]</td>
      <td>61</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Prepare For Town Hall 14! (Clash Of Clans Official)</td>
      <td>[GAMING, CLASHOFCLANS, COC, CLASHOFCLANSGAMEPLAY]</td>
      <td>62</td>
    </tr>
    <tr>
      <th>63</th>
      <td>REVIVED - Derivakat [Dream SMP original song]</td>
      <td>[MUSIC, None, None, None]</td>
      <td>63</td>
    </tr>
    <tr>
      <th>64</th>
      <td>ROCKETS at LAKERS | FULL GAME HIGHLIGHTS | September 6, 2020</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>64</td>
    </tr>
    <tr>
      <th>65</th>
      <td>RUINING All My Favorite Movies With Facts I Didn't Know</td>
      <td>[ENTERTAINMENT, RUININGMOVIESWITHTRIVIA, MYKIE, GLAMANDGORE]</td>
      <td>65</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Rank Strangers Attractiveness | Lineup | Cut</td>
      <td>[ENTERTAINMENT, CUT, WATCHCUT, PEOPLE]</td>
      <td>66</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Reaper Overview | FFXIV Endwalker Media Tour</td>
      <td>[GAMING, FFXIV, FINALFANTASYXIV, FF14]</td>
      <td>67</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Rick and Morty x PlayStation 5 Console [ad]</td>
      <td>[ENTERTAINMENT, ADULTSWIM, ANIMATION, ADULTANIMATION]</td>
      <td>68</td>
    </tr>
    <tr>
      <th>69</th>
      <td>SIEMPRE GIGNAC 💪 | TIGRES 2-1 ULSAN HYUNDAI | MUNDIAL</td>
      <td>[SPORTS, MUNDIALDECLUBES, TIGRES, GIGNAC]</td>
      <td>69</td>
    </tr>
    <tr>
      <th>70</th>
      <td>SPEAKING ONLY CHINESE TO MY FRIENDS FOR 24 HOURS!!</td>
      <td>[PEOPLE&amp;BLOGS, SPEAKINGCHINESETOMYFRIENDS, SPEAKINGONLYCHINESETOMYFRIENDS, SPEAKINGONLYCHINESEFORANENTIREDAY]</td>
      <td>70</td>
    </tr>
    <tr>
      <th>71</th>
      <td>SZA - Good Days (Official Lyric Video)</td>
      <td>[MUSIC, SZA, GOODDAYS, SZAGOODDAYS]</td>
      <td>71</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Samsung should explain this... Exynos vs Qualcomm?! - Note 20 Ultra</td>
      <td>[HOWTO&amp;STYLE, NOTE20ULTRAREVIEW, NOTE20ULTRA, SAMSUNGNOTE20ULTRA]</td>
      <td>72</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Sharon Osbourne Responds To Criticism Over Racism Debate</td>
      <td>[ENTERTAINMENT, ENTERTAINMENT, NEWS, ETCANADA]</td>
      <td>73</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Sounds from the Sideline: Week 5 vs NYG | Dallas Cowboys 2021</td>
      <td>[SPORTS, DALLASCOWBOYS, COWBOYS, CEEDEELAMB]</td>
      <td>74</td>
    </tr>
    <tr>
      <th>75</th>
      <td>SpotemGottem - Sosa Flow (Official Video)</td>
      <td>[MUSIC, SPOTEMGOTTEM, SPOTEMGOTEM, SPOTEMGOTTEM]</td>
      <td>75</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Steelers vs. Ravens Week 8 Highlights | NFL 2020</td>
      <td>[SPORTS, NFL, FOOTBALL, OFFENSE]</td>
      <td>76</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Stephen A. can't see the Nuggets making another 3-1 comeback vs. the Lakers | First Take</td>
      <td>[SPORTS, FIRSTTAKE, NBA, NBAESPN]</td>
      <td>77</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Stephen A. reacts to Giannis leading the Bucks to an NBA championship | First Take</td>
      <td>[SPORTS, ESPN, NBA, SUNSBUCKS]</td>
      <td>78</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Strangers Never Again | Chapter 3</td>
      <td>[ENTERTAINMENT, STRANGERSAGAIN, 2011, STAGES]</td>
      <td>79</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Switching Houses With Alisha Marie!!</td>
      <td>[HOWTO&amp;STYLE, MISSREMIASHTEN, REMLIFE, REMI]</td>
      <td>80</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Tate McRae - rubberband (Official Video)</td>
      <td>[MUSIC, RUBBERBANDTATEMCRAE, TATEMCRAERUBBERBAND, RUBBERBANDSONG]</td>
      <td>81</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Tesla Battery died at very close to charging station #Shorts</td>
      <td>[SCIENCE&amp;TECHNOLOGY, TESLA, ELONMUSK, #SHORTS]</td>
      <td>82</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Toby Keith Happy Birthday America - Fox &amp; Friends Performance</td>
      <td>[FILM&amp;ANIMATION, None, None, None]</td>
      <td>83</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Tory Lanez - Crocodile Teeth Freestyle</td>
      <td>[MUSIC, TORYLANEZ, CROCODILETEETH, SKILLIBENG]</td>
      <td>84</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Turn a Rock into a Heart Pendant</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>85</td>
    </tr>
    <tr>
      <th>86</th>
      <td>U.S. men's 4x400m relay team sets blistering pace to reach final | Tokyo Olympics | NBC Sports</td>
      <td>[SPORTS, OLYMPICS, NBC, NBCSPORTS]</td>
      <td>86</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Usher - Bad Habits (Official Video)</td>
      <td>[MUSIC, USHER, BADHABITS, USHERVEVO]</td>
      <td>87</td>
    </tr>
    <tr>
      <th>88</th>
      <td>WONHO 원호 'OPEN MIND' MV</td>
      <td>[MUSIC, 하이라인, HIGHLINE, 하이라인엔터테인먼트]</td>
      <td>88</td>
    </tr>
    <tr>
      <th>89</th>
      <td>We Tried EXOTIC SNACKS For The FIRST TIME!</td>
      <td>[ENTERTAINMENT, AMPWORLD, AMPWORLDNEWS, BRENTRIVERA]</td>
      <td>89</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Who's Ready For Rugrats 2021? | First Look | Paramount+</td>
      <td>[ENTERTAINMENT, PARAMOUNTPLUS, P+, PARAMOUNT+]</td>
      <td>90</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Whoever Can Survive The Most Days On Their Deserted Island in Hardcore Minecraft Wins</td>
      <td>[GAMING, FORGELABS, FORGELABS, RLCRAFT]</td>
      <td>91</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Why do Corporate Art styles Feel Fake?</td>
      <td>[ENTERTAINMENT, SOLARSANDS, ARTSTYLE, ART]</td>
      <td>92</td>
    </tr>
    <tr>
      <th>93</th>
      <td>With Great Power Comes Great Responsibility...</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>93</td>
    </tr>
    <tr>
      <th>94</th>
      <td>i played the pokémon theme song, but put way too much effort into the video</td>
      <td>[ENTERTAINMENT, SETH, EVERMAN, SETHEVERMAN]</td>
      <td>94</td>
    </tr>
    <tr>
      <th>95</th>
      <td>lazarbeam vs nephew</td>
      <td>[GAMING, None, None, None]</td>
      <td>95</td>
    </tr>
    <tr>
      <th>96</th>
      <td>speed vs adin 1v1 basketball</td>
      <td>[GAMING, None, None, None]</td>
      <td>96</td>
    </tr>
    <tr>
      <th>97</th>
      <td>¡ULTRA GOLAZO! ¡Golazo de Gio! | América 1-0 Chivas | Guard1anes 2020 Liga BBVA MX - J11 | TUDN</td>
      <td>[SPORTS, TELEVISA, UNIVISION, TUDN]</td>
      <td>97</td>
    </tr>
    <tr>
      <th>98</th>
      <td>!@#$%$#!! || Dubov vs Carlsen || Airthings Masters (2020)</td>
      <td>[ENTERTAINMENT, AGADMATOR, CHESS, BESTCHESSCHANNEL]</td>
      <td>98</td>
    </tr>
    <tr>
      <th>99</th>
      <td>'That was my daughter!' Dad yells at suspected shooter who killed teen in 'Forever Purge' showing</td>
      <td>[NEWS&amp;POLITICS, FOREVERPURGE, CORONA, MOVIETHEATER]</td>
      <td>99</td>
    </tr>
    <tr>
      <th>100</th>
      <td>7.5 earthquake hits south Alaska</td>
      <td>[NEWS&amp;POLITICS, EARTHQUAKES, WEATHER, None]</td>
      <td>100</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Amanda Gorman reads a poem at inauguration</td>
      <td>[NEWS&amp;POLITICS, INAUGURATIONDAY, INAUGURATIONDAYPOEM, AMANDAGORMAN]</td>
      <td>101</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Ariana Grande - off the table ft. The Weeknd (Official Live Performance) | Vevo</td>
      <td>[MUSIC, ARIANA, GRANDE, OFF]</td>
      <td>102</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Armed militia, Black Lives Matter protesters collide in Louisville marches</td>
      <td>[NEWS&amp;POLITICS, LOCAL, PROTEST, AMERICANPATRIOTS]</td>
      <td>103</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Ashton Irwin - Skinny Skinny (Official Music Video)</td>
      <td>[MUSIC, ASHTONIRWIN, SKINNYSKINNY, OFFICIALMUSICVIDEO]</td>
      <td>104</td>
    </tr>
    <tr>
      <th>105</th>
      <td>At Home With Kim Kardashian - The End of An Era | Good Morning Vogue</td>
      <td>[ENTERTAINMENT, KEEPINGUPWITHTHEKARDASHIANS, KIMKARDASHIAN, KIMKARDASHIANCELEBRITY]</td>
      <td>105</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Automatic pool stick vs. strangers</td>
      <td>[SCIENCE&amp;TECHNOLOGY, BILLIARDS, POOL, AUTOMATICPOOL]</td>
      <td>106</td>
    </tr>
    <tr>
      <th>107</th>
      <td>BLACK GOKU's speech hit different!</td>
      <td>[FILM&amp;ANIMATION, SSJ9K, SSJCARTER, DBZ]</td>
      <td>107</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Barcelona vs. Ferencváros: Extended Highlights | UCL on CBS</td>
      <td>[SPORTS, BARCELONA, FERENCVÁROSITC, BARCELONAVS.FERENCVÁROSITC]</td>
      <td>108</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Billie Eilish - Getting Older (Official Lyric Video)</td>
      <td>[MUSIC, BILLIE, EILISH, GETTING]</td>
      <td>109</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Brawl Stars: Brawl Talk - Jurassic Splash!</td>
      <td>[GAMING, BRAWLSTARS, MOBILEGAME, MOBILESTRATEGYGAME]</td>
      <td>110</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Brooklyn's 10 DATES in 10 DAYS | Meet Jorge (Date #1)</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>111</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Burna Boy - Level Up (Twice As Tall) (feat. Youssou N'Dour) [Official Audio]</td>
      <td>[MUSIC, REGGAEDANCEHALL, REGGAE, SEKKLEDOWN]</td>
      <td>112</td>
    </tr>
    <tr>
      <th>113</th>
      <td>Corona movie-theater shooting: Vigil honors 2 teens killed in apparent random act | ABC7 LA</td>
      <td>[NEWS&amp;POLITICS, NEWS, None, None]</td>
      <td>113</td>
    </tr>
    <tr>
      <th>114</th>
      <td>DO NOT CHOOSE THE WRONG SECRET BASE! (LUCA, ALBERTO, GUILA!) (Ps3/Xbox360/PS4/XboxOne/PE/MCPE)</td>
      <td>[GAMING, MINECRAFTXBOXONE, MINECRAFTXBOXONESEEDS, MINECRAFTXBOXONESURVIVAL]</td>
      <td>114</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Davido - The Best (Official Video) ft. Mayorkun</td>
      <td>[MUSIC, DAVIDO, DAVIDOTHEBEST, DAVIDOMAYORKUN]</td>
      <td>115</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Deaths reported at Kabul airport as Afghans try to flee Taliban - BBC News</td>
      <td>[NEWS&amp;POLITICS, BBC, BBCNEWS, NEWS]</td>
      <td>116</td>
    </tr>
    <tr>
      <th>117</th>
      <td>ESCAPE ROOM: TOURNAMENT OF CHAMPIONS - Official Trailer (HD) | In Theaters July 16</td>
      <td>[ENTERTAINMENT, ESCAPEROOM, ESCAPEROOMMOVIE, ESCAPEROOMHD]</td>
      <td>117</td>
    </tr>
    <tr>
      <th>118</th>
      <td>Extreme Pantry Organization | Organizing Vlog</td>
      <td>[PEOPLE&amp;BLOGS, ORGANIZINGVLOG, PANTRYORGANIZATION, EXTREMEORGANIZATION]</td>
      <td>118</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Game Theory: Are Your Mobile Games ILLEGAL?</td>
      <td>[GAMING, FAKEMOBILEGAMES, FAKEGAMES, MOBILEGAMES]</td>
      <td>119</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Gol D Roger vs Whitebeard | One Piece</td>
      <td>[FILM&amp;ANIMATION, CRUNCHYROLL, ANIME, ANIMETRAILER]</td>
      <td>120</td>
    </tr>
    <tr>
      <th>121</th>
      <td>Guess that WORD and I'll BUY It Challenge w/ 2HYPE</td>
      <td>[SPORTS, FAMILYFRIENDLYYOUTUBERSCURSE, IPHONE11PROMAXUNBOXING, IPHONE11PROUNBOXING]</td>
      <td>121</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Guess the Drawing with Taylor Alesia #shorts</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>122</td>
    </tr>
    <tr>
      <th>123</th>
      <td>Gwen: The Hallowed Seamstress | Champion Gameplay Trailer - League of Legends</td>
      <td>[GAMING, RIOTGAMES, RIOT, LEAGUEOFLEGENDS]</td>
      <td>123</td>
    </tr>
    <tr>
      <th>124</th>
      <td>HAZBIN HOTEL + HELLUVA BOSS HALLOWEEN SPOOKY SALE (+ SPECIAL UPDATE!)</td>
      <td>[FILM&amp;ANIMATION, VIVZIEPOP, ZOOPHOBIA, None]</td>
      <td>124</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Harry Kane sends England to Euro 2020 final after extra-time win | Highlights | ESPN FC</td>
      <td>[SPORTS, ENGLANDVSDENMARK, ENGLANDDENMARK, ENGLAND]</td>
      <td>125</td>
    </tr>
    <tr>
      <th>126</th>
      <td>Highlights | España 6-0 Alemania | UEFA Nations League - J6 | TUDN</td>
      <td>[SPORTS, UNIVISION, DEPORTES, TUDN]</td>
      <td>126</td>
    </tr>
    <tr>
      <th>127</th>
      <td>How to Make a Toblerone Latte #shorts</td>
      <td>[FILM&amp;ANIMATION, None, None, None]</td>
      <td>127</td>
    </tr>
    <tr>
      <th>128</th>
      <td>How to Waste $517.23 on Amazon...</td>
      <td>[SCIENCE&amp;TECHNOLOGY, MYSTERYTECH, TECH, UNBOXING]</td>
      <td>128</td>
    </tr>
    <tr>
      <th>129</th>
      <td>I Found The SECRET CUBE PLANET... (and blew it up...) | Solar Smash</td>
      <td>[GAMING, KINDLYKEYIN, KEYIN, SOLARSMASH]</td>
      <td>129</td>
    </tr>
    <tr>
      <th>130</th>
      <td>I expected one, but not two 😆 | Ronaldo speaks after homecoming brace!</td>
      <td>[SPORTS, SKYSPORTS, PREMIERLEAGUE, FOOTBALLLEAGUE]</td>
      <td>130</td>
    </tr>
    <tr>
      <th>131</th>
      <td>I spent a day with KIDNAPPING SURVIVORS</td>
      <td>[EDUCATION, ANTHONYPADILLA, PADILLA, ANTHONY]</td>
      <td>131</td>
    </tr>
    <tr>
      <th>132</th>
      <td>If MRBEAST lived in a Mexican House</td>
      <td>[COMEDY, MRBEASTINAMEXICANHUOSE, IFMRBEASTLIVEDINAMEXICANHOUSE, MEXICANHOME]</td>
      <td>132</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Japón vs México 2-1 | Fútbol Masculino | Tokyo 2020 | Telemundo Deportes</td>
      <td>[SPORTS, TELEMUNDODEPORTES, MÉXICO, JAPÓN]</td>
      <td>133</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Just a General Update Video.</td>
      <td>[AUTOS&amp;VEHICLES, THESTRADMAN, STRADMAN, SUPERCARS]</td>
      <td>134</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Katy Perry - Not the End of the World</td>
      <td>[MUSIC, KATY, PERRY, NOT]</td>
      <td>135</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Lapiz Conciente - 9 Dias</td>
      <td>[MUSIC, None, None, None]</td>
      <td>136</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Liam Payne, Dixie D’Amelio - Naughty List</td>
      <td>[MUSIC, LIAM, PAYNE, DIXIE]</td>
      <td>137</td>
    </tr>
    <tr>
      <th>138</th>
      <td>Lionel Messi puts home unbelievable free-kick goal vs. Chile | 2021 Copa América Highlights</td>
      <td>[SPORTS, FOX, FOXSPORTS, SOCCER]</td>
      <td>138</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Machine Gun Kelly, Halsey, blackbear - Downfalls High ft. Trippie Redd, iann dior</td>
      <td>[MUSIC, MACHINE, GUN, KELLY]</td>
      <td>139</td>
    </tr>
    <tr>
      <th>140</th>
      <td>Melanie Martinez - Notebook [Official Audio]</td>
      <td>[MUSIC, MELANIEMARTINEZ, MELANIE, MARTINEZ]</td>
      <td>140</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Michael B. Jordan Is 'Struggling' After Chadwick Boseman's Death</td>
      <td>[ENTERTAINMENT, MICHAELBJORDAN, CHADWICKBOSEMAN, BLACKPANTHER]</td>
      <td>141</td>
    </tr>
    <tr>
      <th>142</th>
      <td>Michael Becomes Jim - The Office US</td>
      <td>[ENTERTAINMENT, THEOFFICE, THEOFFICEFULLEPISODES, RAINNWILSON]</td>
      <td>142</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Minecraft Live: The Nether Update Encore</td>
      <td>[GAMING, MINECRAFT, LIVE, MINECRAFTLIVE]</td>
      <td>143</td>
    </tr>
    <tr>
      <th>144</th>
      <td>Minecraft Speedrunner VS 4 Hunters FINALE</td>
      <td>[GAMING, DREAMMINECRAFT, DREAMMINECRAFTYOUTUBE, MINECRAFT]</td>
      <td>144</td>
    </tr>
    <tr>
      <th>145</th>
      <td>Minecraft, But It's Only 1 Custom Block...</td>
      <td>[GAMING, MINECRAFT, MINECRAFTBUTIT'SONLY1CUSTOMBLOCK, MINECRAFTBUT]</td>
      <td>145</td>
    </tr>
    <tr>
      <th>146</th>
      <td>Moneybagg Yo – Said Sum Remix feat. City Girls, DaBaby [Official Music Video]</td>
      <td>[MUSIC, MONEYBAGGYOSAIDSUM, MONEYBAGG, MONEYBAG]</td>
      <td>146</td>
    </tr>
    <tr>
      <th>147</th>
      <td>Mora x Bad Bunny x Sech - Volando Remix (Video Oficial)</td>
      <td>[MUSIC, MORA, BADBUNNY, SECH]</td>
      <td>147</td>
    </tr>
    <tr>
      <th>148</th>
      <td>Mulan - Movie Review</td>
      <td>[ENTERTAINMENT, MULAN, MOVIEREVIEW, 2020]</td>
      <td>148</td>
    </tr>
    <tr>
      <th>149</th>
      <td>NBA 2K21: Next-Gen Gameplay + Developer Commentary</td>
      <td>[SPORTS, None, None, None]</td>
      <td>149</td>
    </tr>
    <tr>
      <th>150</th>
      <td>Omy De Oro, Alex Rose - No Te Asustes Remix feat. Jay Wheeler, Miky Woodz (ERDP)</td>
      <td>[MUSIC, ERDP, ELREYDELPUNCHLINE, OMYDEORO]</td>
      <td>150</td>
    </tr>
    <tr>
      <th>151</th>
      <td>Opposite Twins Shop for Eachother on Amazon (actually cute)</td>
      <td>[HOWTO&amp;STYLE, OPPOSITETWINSSHOPFOREACHOTHERONAMAZONCHALLENGE, SHOPPINGCHALLENGE, NIKIANDGABI]</td>
      <td>151</td>
    </tr>
    <tr>
      <th>152</th>
      <td>PICKING Up My FRIENDS As An UNDERCOVER UBER DRIVER!!</td>
      <td>[ENTERTAINMENT, UBER, UNDERCOVER, UNDERCOVERUBER]</td>
      <td>152</td>
    </tr>
    <tr>
      <th>153</th>
      <td>President-Elect Joe Biden Delivers Remarks As Trump Supporters Storm U.S. Capitol | TIME</td>
      <td>[NEWS&amp;POLITICS, TIME, TIMEMAGAZINE, MAGAZINE]</td>
      <td>153</td>
    </tr>
    <tr>
      <th>154</th>
      <td>Randy Orton ends up in Alexa’s Playground: Raw, Dec. 21, 2020</td>
      <td>[SPORTS, WWE, WORLDWRESTLINGENTERTAINMENT, WRESTLING]</td>
      <td>154</td>
    </tr>
    <tr>
      <th>155</th>
      <td>SHOCKING WAR FACTS #shorts</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>155</td>
    </tr>
    <tr>
      <th>156</th>
      <td>SPIDERMAN No Way Home Official Trailer Breakdown | Easter Eggs Explained &amp; Things You Missed</td>
      <td>[ENTERTAINMENT, SPIDERMANNOWAYHOMETRAILER, SPIDERMAN, SPIDERMANNOWAYHOMETRAILERBREAKDOWN]</td>
      <td>156</td>
    </tr>
    <tr>
      <th>157</th>
      <td>Seas #TeamSeas</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>157</td>
    </tr>
    <tr>
      <th>158</th>
      <td>Second 2020 Presidential Debate between Donald Trump and Joe Biden</td>
      <td>[NEWS&amp;POLITICS, C-SPAN, CSPAN, 2020]</td>
      <td>158</td>
    </tr>
    <tr>
      <th>159</th>
      <td>Space Jam: A New Legacy Review - YMS</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>159</td>
    </tr>
    <tr>
      <th>160</th>
      <td>Stephen Sharer - SLUSHIE (Official Music Video)</td>
      <td>[MUSIC, STEPHENSHARER, SHARETHELOVE, STEVENSHARE]</td>
      <td>160</td>
    </tr>
    <tr>
      <th>161</th>
      <td>The Gate Guardian | SCP-001 (SCP Animation)</td>
      <td>[FILM&amp;ANIMATION, SCPANIMATED, SCP, TALESFROMTHEFOUNDATION]</td>
      <td>161</td>
    </tr>
    <tr>
      <th>162</th>
      <td>The Haunting of Shane Dawson</td>
      <td>[COMEDY, SHANE, DAWSON, None]</td>
      <td>162</td>
    </tr>
    <tr>
      <th>163</th>
      <td>The Scooby Doo Movies Are Insane</td>
      <td>[COMEDY, DREWGOODEN, COMEDY, COMMENTARY]</td>
      <td>163</td>
    </tr>
    <tr>
      <th>164</th>
      <td>The Supernova That Measured The Universe</td>
      <td>[EDUCATION, VERITASIUM, SCIENCE, PHYSICS]</td>
      <td>164</td>
    </tr>
    <tr>
      <th>165</th>
      <td>This week on RELEASED | ROSÉ (Official Trailer)</td>
      <td>[MUSIC, NEWMUSIC, RELEASEDYOUTUBEORIGINALS, YOUTUBEORIGINALS]</td>
      <td>165</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Tommy Fury reacts to Jake Paul altercation immediately after Tyron Woodley bout &amp; Anthony Taylor win</td>
      <td>[SPORTS, BOXING, INTERVIEW, BOXINGNEWS]</td>
      <td>166</td>
    </tr>
    <tr>
      <th>167</th>
      <td>True Facts: The Hummingbird Warrior</td>
      <td>[PETS&amp;ANIMALS, None, None, None]</td>
      <td>167</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Turns out Tubbo is Cracked at Dodgeball</td>
      <td>[GAMING, WILBURSOOT, LIVE, WILBURLIVE]</td>
      <td>168</td>
    </tr>
    <tr>
      <th>169</th>
      <td>Uncle Roger Think Cowboy Fried Rice SO WEIRD (Kent Rollins)</td>
      <td>[COMEDY, NIGELNG, UNCLEROGER, NIGELNGCOMEDY]</td>
      <td>169</td>
    </tr>
    <tr>
      <th>170</th>
      <td>Welcome To The World</td>
      <td>[PEOPLE&amp;BLOGS, ROMAN, ATWOOD, THEATWOODS]</td>
      <td>170</td>
    </tr>
    <tr>
      <th>171</th>
      <td>When guys try too hard to be cool.</td>
      <td>[COMEDY, None, None, None]</td>
      <td>171</td>
    </tr>
    <tr>
      <th>172</th>
      <td>Where is America?</td>
      <td>[COMEDY, JAYFOREMAN, MARKCOOPER-JONES, MAPMEN]</td>
      <td>172</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Why I REGRET buying my Hummer H1 (I'm so sorry)</td>
      <td>[AUTOS&amp;VEHICLES, None, None, None]</td>
      <td>173</td>
    </tr>
    <tr>
      <th>174</th>
      <td>Will Sarah Impress The Sharks With Her Fire Hose Fitness Tool? | Shark Tank US</td>
      <td>[PEOPLE&amp;BLOGS, SHARKTANK, SHARKTANKUS, SHARKTANKUSA]</td>
      <td>174</td>
    </tr>
    <tr>
      <th>175</th>
      <td>YUNGBLUD with Denzel Curry - Lemonade</td>
      <td>[MUSIC, YUNGBLUD, YUNGBLUD, YOUNGBLOOD]</td>
      <td>175</td>
    </tr>
    <tr>
      <th>176</th>
      <td>[CHOREOGRAPHY] BTS (방탄소년단) 'Butter' Dance Practice</td>
      <td>[MUSIC, 방탄소년단, BTS, BANGTAN]</td>
      <td>176</td>
    </tr>
    <tr>
      <th>177</th>
      <td>“Change of Plans” | Marvel Studios’ Hawkeye | Disney+</td>
      <td>[ENTERTAINMENT, MARVEL, COMICS, None]</td>
      <td>177</td>
    </tr>
    <tr>
      <th>178</th>
      <td>'Black Panther' Star Chadwick Boseman Dies Of Cancer At Age 43</td>
      <td>[NEWS&amp;POLITICS, KCAL9NEWSEVENING, CHADWICKBOSEMAN, BLACKPANTHER]</td>
      <td>178</td>
    </tr>
    <tr>
      <th>179</th>
      <td>'One Of My Constituents Got This In The Mail': John Kennedy Presents Surprising Letter At Hearing</td>
      <td>[NEWS&amp;POLITICS, SEN.JOHNKENNEDY, LABORSEC.MARTYWALSH, SENATE]</td>
      <td>179</td>
    </tr>
    <tr>
      <th>180</th>
      <td>*EXCITING Renovation Update* I have so much to show you!</td>
      <td>[PEOPLE&amp;BLOGS, RENOVATION, HOUSERENO, DREAMHOUSE]</td>
      <td>180</td>
    </tr>
    <tr>
      <th>181</th>
      <td>*NEW* BACKYARDIGANS SCARY Mod in Among Us</td>
      <td>[GAMING, None, None, None]</td>
      <td>181</td>
    </tr>
    <tr>
      <th>182</th>
      <td>*intense* room transformation</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>182</td>
    </tr>
    <tr>
      <th>183</th>
      <td>A DAY IN THE LIFE OF OFFLINETV</td>
      <td>[GAMING, OFFLINETV, SCARRA, POKI]</td>
      <td>183</td>
    </tr>
    <tr>
      <th>184</th>
      <td>AC/DC - Through The Mists Of Time (Official Video)</td>
      <td>[MUSIC, ACDC, MISTSOFTIME, THROUGHTHEMISTSOFTIME]</td>
      <td>184</td>
    </tr>
    <tr>
      <th>185</th>
      <td>Adele - Easy On Me (Clip)</td>
      <td>[MUSIC, ADELE, EASYONME, EOM]</td>
      <td>185</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Apologizing for my last video</td>
      <td>[EDUCATION, FOOTDOCDANA, FOOTDOCDANA, TIKTOKSSURGEON]</td>
      <td>186</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Barb &amp; Star Go To Vista Del Mar (2021 Movie) Official Trailer – Kristen Wiig, Annie Mumolo</td>
      <td>[FILM&amp;ANIMATION, BARBANDSTARGOTOVISTADELMAR, BARBANDSTAR, BARBANDSTARGOTOVISTADELMARMOVIE]</td>
      <td>187</td>
    </tr>
    <tr>
      <th>188</th>
      <td>BeckSeat Driver ft. Charli, Dixie, James, Larray, &amp; Chase</td>
      <td>[ENTERTAINMENT, HALLOWEEN, BECKSEATDRIVER, BECK]</td>
      <td>188</td>
    </tr>
    <tr>
      <th>189</th>
      <td>Billie Eilish: Same Interview, The Fourth Year | Vanity Fair</td>
      <td>[ENTERTAINMENT, BILLIEEILISH, SAMEINTERVIEW, SAMEINTERVIEWFOURTHYEAR]</td>
      <td>189</td>
    </tr>
    <tr>
      <th>190</th>
      <td>Bishop Sycamore - The High School That Isn't</td>
      <td>[SPORTS, BISHOPSYCAMOREOHIO, BISHOPSYCAMORE, BISHOPSYCAMORE:THEHIGHSCHOOLTHATISN'T]</td>
      <td>190</td>
    </tr>
    <tr>
      <th>191</th>
      <td>Booka600 ft. Lil Durk - Relentless (Official Video)</td>
      <td>[MUSIC, BOOKA600, BOOKA600OFFICIAL, BOOKA600MUSIC]</td>
      <td>191</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Brendan Schaub on a Horrific 18-Wheeler Family Freeway Accident | GoFundMe in Description</td>
      <td>[COMEDY, THEFIGHTERANDTHEKID, KID, THEFIGHTER]</td>
      <td>192</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Clash Royale: Enter The Forbidden Palace! First Elite Barbarian Emotes! (New Season!)</td>
      <td>[GAMING, CLASHROYALE, CLASHROYALEGAME, SUPERCELL]</td>
      <td>193</td>
    </tr>
    <tr>
      <th>194</th>
      <td>DUALITY // Official Lore Cinematic - VALORANT</td>
      <td>[GAMING, VALORANT, DUALITY, CINEMATIC]</td>
      <td>194</td>
    </tr>
    <tr>
      <th>195</th>
      <td>David Benavidez vs. Alexis Angulo: Highlights | SHOWTIME CHAMPIONSHIP BOXING</td>
      <td>[SPORTS, SHOWTIME, SHOSPORTS, SPORTS]</td>
      <td>195</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Did We Just Detect Life on Venus?</td>
      <td>[SCIENCE&amp;TECHNOLOGY, VENUS, LIFEONVENUS, LIFEDETECTEDVENUS]</td>
      <td>196</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Everyday Objects But MINI</td>
      <td>[ENTERTAINMENT, SSSNIPERWOLF, SNIPERWOLF, REACTING]</td>
      <td>197</td>
    </tr>
    <tr>
      <th>198</th>
      <td>FC Porto vs. Chelsea: Extended Highlights | UCL on CBS Sports</td>
      <td>[SPORTS, CHELSEA, FCPORTO, FCPORTOVS.CHELSEA]</td>
      <td>198</td>
    </tr>
    <tr>
      <th>199</th>
      <td>FNAF Security Breach OFFICIAL TRAILER (FNAF 9)</td>
      <td>[GAMING, FNAFSECURITYBREACH, FNAFSECURITYBREACHTRAILER, FNAFSECURITYBREACHGAMEPLAY]</td>
      <td>199</td>
    </tr>
    <tr>
      <th>200</th>
      <td>First Problem With The 1000HP Lamborghini!</td>
      <td>[ENTERTAINMENT, TANNERFOX, TFOX, VLOG]</td>
      <td>200</td>
    </tr>
    <tr>
      <th>201</th>
      <td>FlightReacts Plays NBA 2K21 For The FIRST Time &amp; This HAPPENED!</td>
      <td>[GAMING, NBA2K202KNBA, NBA2K20, NBA2K20BESTBUILD]</td>
      <td>201</td>
    </tr>
    <tr>
      <th>202</th>
      <td>Flipp Dinero ft. A Boogie Wit Da Hoodie - No No No (Official Music Video)</td>
      <td>[MUSIC, FLIPP, FLIPPDINERO, GUALA]</td>
      <td>202</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Fortnite Raptors Are Hatching Across the Island</td>
      <td>[GAMING, FORTNITERAPTORS, RAPTORS, FORTNITEPRIMAL]</td>
      <td>203</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Fudgy, the Golden Retriever Puppy vs Slow Feeder</td>
      <td>[TRAVEL&amp;EVENTS, SIMON, TOKYO, JAPAN]</td>
      <td>204</td>
    </tr>
    <tr>
      <th>205</th>
      <td>GMC HUMMER EV | “Revolutionary World Premiere” | GMC</td>
      <td>[AUTOS&amp;VEHICLES, None, None, None]</td>
      <td>205</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Galaxy x BTS: A Piece of Cake 🍰 | Samsung</td>
      <td>[SCIENCE&amp;TECHNOLOGY, SAMSUNG, SAMSUNGGALAXY, GALAXY]</td>
      <td>206</td>
    </tr>
    <tr>
      <th>207</th>
      <td>Gervonta Davis vs Mario Barrios Knockout HIGHLIGHTS: June 26, 2021 - PBC on Showtime PPV</td>
      <td>[SPORTS, BOXING, PBC, PREMIERBOXINGCHAMPIONS]</td>
      <td>207</td>
    </tr>
    <tr>
      <th>208</th>
      <td>Giants vs. Seahawks Week 13 Highlights | NFL 2020</td>
      <td>[SPORTS, NFL, FOOTBALL, OFFENSE]</td>
      <td>208</td>
    </tr>
    <tr>
      <th>209</th>
      <td>He officially quit</td>
      <td>[ENTERTAINMENT, FAZERUG, RUG, RUGFAZE]</td>
      <td>209</td>
    </tr>
    <tr>
      <th>210</th>
      <td>Hermitcraft 8: A New Hermit has Arrived! Episode 1</td>
      <td>[GAMING, MINECRAFT, GEMINITAY, HERMITCRAFT]</td>
      <td>210</td>
    </tr>
    <tr>
      <th>211</th>
      <td>Hide And Seek Across The Earth!</td>
      <td>[GAMING, None, None, None]</td>
      <td>211</td>
    </tr>
    <tr>
      <th>212</th>
      <td>How I Animated This Video</td>
      <td>[FILM&amp;ANIMATION, JOELHAVER, ANIMATION, TUTORIAL]</td>
      <td>212</td>
    </tr>
    <tr>
      <th>213</th>
      <td>I Cheated with a FAKE Professional Builder in a Building Competition...</td>
      <td>[GAMING, MINECRAFT, MINECRAFT, MINECRAFTYOUTUBER]</td>
      <td>213</td>
    </tr>
    <tr>
      <th>214</th>
      <td>I Got Backstabbed...</td>
      <td>[ENTERTAINMENT, KSI, KSIOLAJIDEBT, KSIOLAJIDEBTHD]</td>
      <td>214</td>
    </tr>
    <tr>
      <th>215</th>
      <td>I bought $1000 worth of Bootleg merch</td>
      <td>[COMEDY, BOOTLEGMERCH, MERCH, ANIMATION]</td>
      <td>215</td>
    </tr>
    <tr>
      <th>216</th>
      <td>I'M LAUNCHING A YOUTUBE CHANNEL by Van Neistat</td>
      <td>[FILM&amp;ANIMATION, 4-18-16, None, None]</td>
      <td>216</td>
    </tr>
    <tr>
      <th>217</th>
      <td>International Burger King Taste Test</td>
      <td>[ENTERTAINMENT, GMM, GOODMYTHICALMORNING, RHETTANDLINK]</td>
      <td>217</td>
    </tr>
    <tr>
      <th>218</th>
      <td>Introducing Fudgy the Golden Retriever Puppy</td>
      <td>[PETS&amp;ANIMALS, SIMONANDMARTINA, SIMON, MARTINA]</td>
      <td>218</td>
    </tr>
    <tr>
      <th>219</th>
      <td>Justin Bieber - Anyone</td>
      <td>[MUSIC, JUSTIN, BIEBER, ANYONE]</td>
      <td>219</td>
    </tr>
    <tr>
      <th>220</th>
      <td>King Von - Wayne's Story (Official Video)</td>
      <td>[MUSIC, KINGVON, KINGVON2018, KINGVONOFFICIAL]</td>
      <td>220</td>
    </tr>
    <tr>
      <th>221</th>
      <td>Lamb | Official Trailer HD | A24</td>
      <td>[FILM&amp;ANIMATION, A24, A24FILMS, A24TRAILERS]</td>
      <td>221</td>
    </tr>
    <tr>
      <th>222</th>
      <td>Lil Baby - Behind The Scenes of “The Bigger Picture” at The 63rd GRAMMYs</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>222</td>
    </tr>
    <tr>
      <th>223</th>
      <td>Loki Episode 4 Breakdown! PROOF KANG IS VILLAIN! I Figured It Out!?</td>
      <td>[FILM&amp;ANIMATION, LOKIEPISODE4BREAKDOWN, LOKIEPISODE4ENDINGEXPLAINED, LOKIEPISODE4MIDCREDITSEXPLAINED]</td>
      <td>223</td>
    </tr>
    <tr>
      <th>224</th>
      <td>Lovejoy - Pebble Brain (Full EP)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>224</td>
    </tr>
    <tr>
      <th>225</th>
      <td>Making 2,163 Potions To Kill This Entire Minecraft Server...</td>
      <td>[GAMING, ROSHAMBOGAMES, MINECRAFT, LIFESTEALSMP]</td>
      <td>225</td>
    </tr>
    <tr>
      <th>226</th>
      <td>Maluma, Jennifer Lopez - Lonely (Official Video)</td>
      <td>[MUSIC, JLO, JENNIFER, LOPEZ]</td>
      <td>226</td>
    </tr>
    <tr>
      <th>227</th>
      <td>Messi and his Kids on their way to the pitch</td>
      <td>[SPORTS, FOOTBALL, FUTBOL, BEINSPORTS]</td>
      <td>227</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Minecraft but you can Mine mobs</td>
      <td>[GAMING, MINECRAFT, NEWITEMSINMINECRAFT, NEWWEAPONS]</td>
      <td>228</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Morgan Wallen - Still Goin Down (Official Lyric Video)</td>
      <td>[MUSIC, MORGAN, WALLEN, STILL]</td>
      <td>229</td>
    </tr>
    <tr>
      <th>230</th>
      <td>My Girlfriend, My Best Friend and the Barfy Beach Date</td>
      <td>[COMEDY, GIRLFRIEND, GIF, FLOOF]</td>
      <td>230</td>
    </tr>
    <tr>
      <th>231</th>
      <td>My Pregnancy Story (Why I Kept It a Secret…)</td>
      <td>[HOWTO&amp;STYLE, HELLOHUNNAY, JEANNIEMAI, KIN]</td>
      <td>231</td>
    </tr>
    <tr>
      <th>232</th>
      <td>Nas - Nobody feat. Ms. Lauryn Hill (Official Audio)</td>
      <td>[MUSIC, NAS, KING'SDISEASE2, KING'SDISEASE]</td>
      <td>232</td>
    </tr>
    <tr>
      <th>233</th>
      <td>PANTON SQUAD NEW HOUSE  TOUR</td>
      <td>[ENTERTAINMENT, FAMILYFRIENDLY, PANTONSQUAD, FAMILY]</td>
      <td>233</td>
    </tr>
    <tr>
      <th>234</th>
      <td>Patrice Evra breaks down into passionate rant after Man Utd's 6-1 defeat to Spurs</td>
      <td>[SPORTS, SKYSPORTS, SKYSPORTSFOOTBALL, PREMIERLEAGUE]</td>
      <td>234</td>
    </tr>
    <tr>
      <th>235</th>
      <td>Pelosi: Proposing a stand-alone bill for $2,000 payments on Monday</td>
      <td>[NEWS&amp;POLITICS, VIDEO, PELOSI:PROPOSINGASTAND-ALONEBILLFOR$2000PAYMENTSONMONDAY, None]</td>
      <td>235</td>
    </tr>
    <tr>
      <th>236</th>
      <td>People Having A Worse Day Than You</td>
      <td>[ENTERTAINMENT, SSSNIPERWOLF, SNIPERWOLF, REACTING]</td>
      <td>236</td>
    </tr>
    <tr>
      <th>237</th>
      <td>Pokémon Shield, but I Randomized Everything</td>
      <td>[GAMING, MANDJTV, MANDJTVPOKEVIDS, GAMEPLAY]</td>
      <td>237</td>
    </tr>
    <tr>
      <th>238</th>
      <td>Quadeca Reacts to the KSI ALBUM...</td>
      <td>[PEOPLE&amp;BLOGS, KSIQUADECA, ALBUMN, REACTION]</td>
      <td>238</td>
    </tr>
    <tr>
      <th>239</th>
      <td>ROBIN Test Footage Scene</td>
      <td>[ENTERTAINMENT, JAMIECOSTA, JAMIE, COSTA]</td>
      <td>239</td>
    </tr>
    <tr>
      <th>240</th>
      <td>Rylo Rodriguez - We Could Never Die (Official Video)</td>
      <td>[ENTERTAINMENT, RYLO, RYLORODRIGUEZ, WECOULDNEVERDIE]</td>
      <td>240</td>
    </tr>
    <tr>
      <th>241</th>
      <td>SIDEMEN $20,000 A-Z EATING CHALLENGE</td>
      <td>[ENTERTAINMENT, SIDEMEN, SIDEMENSUNDAY, #SIDEMENSUNDAY]</td>
      <td>241</td>
    </tr>
    <tr>
      <th>242</th>
      <td>Someone Is Selling My 50 MIL Award On Ebay</td>
      <td>[ENTERTAINMENT, SATIRE, None, None]</td>
      <td>242</td>
    </tr>
    <tr>
      <th>243</th>
      <td>SpaceX Starship SN11 Analysis</td>
      <td>[SCIENCE&amp;TECHNOLOGY, GRAVITYLINKSTARSHIP, SPINGRAVITY, ARTIFICIALGRAVITY]</td>
      <td>243</td>
    </tr>
    <tr>
      <th>244</th>
      <td>Speed Bridge Kill Skywars Servers #Shorts</td>
      <td>[GAMING, SKYWARSSERVERS, MINECRAFTSKYWARSSERVERS, HYPIXELSKYWARS]</td>
      <td>244</td>
    </tr>
    <tr>
      <th>245</th>
      <td>Spinning an Apple until it Explodes at 28,500fps - The Slow Mo Guys</td>
      <td>[ENTERTAINMENT, SLOMO, SLOW, MO]</td>
      <td>245</td>
    </tr>
    <tr>
      <th>246</th>
      <td>Starbase Launchpad Tour with Elon Musk [PART 3]</td>
      <td>[SCIENCE&amp;TECHNOLOGY, ELONMUSKINTERVIEW, ELONMUSKSTARBASETOUR, ELONMUSKSTARBASEFACTORY]</td>
      <td>246</td>
    </tr>
    <tr>
      <th>247</th>
      <td>Surviving 100 Days in a Minecraft WAR.. here's what happened</td>
      <td>[GAMING, MINECRAFT, RYANNOTBRIAN, 100DAYS]</td>
      <td>247</td>
    </tr>
    <tr>
      <th>248</th>
      <td>THIS IS WHAT REALLY HAPPENED!!! **OFFICIAL VIDEO**</td>
      <td>[PEOPLE&amp;BLOGS, THISISWHATREALLYHAPPENED, THEACEFAMILYTHISISWHATREALLYHAPPENED, ACEFAMILYTHISISWHATREALLYHAPPENED]</td>
      <td>248</td>
    </tr>
    <tr>
      <th>249</th>
      <td>Testing the *BEST* Kitchen Gadgets EVER!? - Part 8</td>
      <td>[HOWTO&amp;STYLE, KITCHEN, GADGETS, AMAZON]</td>
      <td>249</td>
    </tr>
    <tr>
      <th>250</th>
      <td>The NBA Arrives In Fortnite</td>
      <td>[GAMING, YT:CC=ON, FORTNITE, FORTNITENBA]</td>
      <td>250</td>
    </tr>
    <tr>
      <th>251</th>
      <td>The Story of Minecraft</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>251</td>
    </tr>
    <tr>
      <th>252</th>
      <td>Tom Brady on Return to New England, Breaking All-Time Passing Yards Record | Press Conference</td>
      <td>[SPORTS, PRESSCONFERENCE, BRUCEARIANS, TOMBRADY]</td>
      <td>252</td>
    </tr>
    <tr>
      <th>253</th>
      <td>Trope Talk: Loners</td>
      <td>[EDUCATION, FUNNY, SUMMARY, OSP]</td>
      <td>253</td>
    </tr>
    <tr>
      <th>254</th>
      <td>Trying durian as an adult</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>254</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Twenty One Pilots - No Chances (Lyric Video)</td>
      <td>[MUSIC, TWENTYONEPILOTSNOCHANCES, NOCHANCESLYRICS, TWENTYONEPILOTSLYRICS]</td>
      <td>255</td>
    </tr>
    <tr>
      <th>256</th>
      <td>Undercover Boss is Ridiculous</td>
      <td>[COMEDY, None, None, None]</td>
      <td>256</td>
    </tr>
    <tr>
      <th>257</th>
      <td>Ups &amp; Downs From AEW All Out 2021</td>
      <td>[SPORTS, WWE, WRESTLING, WHATCULTURE]</td>
      <td>257</td>
    </tr>
    <tr>
      <th>258</th>
      <td>VENOM: LET THERE BE CARNAGE TRAILER REACTION!! (Venom 2 | Official)</td>
      <td>[ENTERTAINMENT, VENOM2TRAILER, VENOM:LETTHEREBECARNAGE-OFFICIALTRAILER, VENOM2TRAILERREACTION]</td>
      <td>258</td>
    </tr>
    <tr>
      <th>259</th>
      <td>VFX Artists React to Bad &amp; Great CGi 49</td>
      <td>[ENTERTAINMENT, VFX, CGI, VFXARTISTSREACT]</td>
      <td>259</td>
    </tr>
    <tr>
      <th>260</th>
      <td>WALL STREET LOSSES! - The TRUTH Behind GameStop, WallStreetBets &amp; Robinhood | Kevin O'Leary</td>
      <td>[EDUCATION, KEVINO'LEARY, KEVINO'LEARYINTERVIEW, KEVINO'LEARYSPEECH]</td>
      <td>260</td>
    </tr>
    <tr>
      <th>261</th>
      <td>We have some explaining to do...</td>
      <td>[PEOPLE&amp;BLOGS, BRAWADIS, PRANK, BASKETBALL]</td>
      <td>261</td>
    </tr>
    <tr>
      <th>262</th>
      <td>Which of these clubs would Lionel Messi leave Barcelona for? ► 442oons</td>
      <td>[SPORTS, FOOTBALL, SOCCER, ONEFOOTBALL]</td>
      <td>262</td>
    </tr>
    <tr>
      <th>263</th>
      <td>YoungBoy Never Broke Again - Nevada [Official Audio]</td>
      <td>[MUSIC, YOUNGBOYNEVERBROKEAGAIN, NBAYOUNGBOY, YOUNGBOYIGLIVE]</td>
      <td>263</td>
    </tr>
    <tr>
      <th>264</th>
      <td>Yungeen Ace - Opp Boyz (Official Music Video)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>264</td>
    </tr>
    <tr>
      <th>265</th>
      <td>[최초공개] ENHYPEN(엔하이픈) - Not For Sale (4K) | ENHYPEN COMEBACK SHOW 'CARNIVAL' | Mnet 210426 방송</td>
      <td>[ENTERTAINMENT, 엠넷, MNET, 엠투]</td>
      <td>265</td>
    </tr>
    <tr>
      <th>266</th>
      <td>how class discussions went in middle school</td>
      <td>[COMEDY, HOWCLASSROOMDISCUSSIONSWENTINMIDDLESCHOOL, MIDDLESCHOOLGUS, GUS]</td>
      <td>266</td>
    </tr>
    <tr>
      <th>267</th>
      <td>the rare male karen</td>
      <td>[ENTERTAINMENT, SUBWAY, TIKTOK, STORY]</td>
      <td>267</td>
    </tr>
    <tr>
      <th>268</th>
      <td>#1 Alabama Crimson Tide vs. LSU Tigers: Extended Highlights| CBS Sports HQ</td>
      <td>[SPORTS, ALABAMACRIMSONTIDE, LSUTIGERS, ALABAMACRIMSONTIDEVS.LSUTIGERS]</td>
      <td>268</td>
    </tr>
    <tr>
      <th>269</th>
      <td>$100,000 Golf CHALLENGE with MMG, TJass, Jesser and Kris London | HoH Showdown</td>
      <td>[SPORTS, None, None, None]</td>
      <td>269</td>
    </tr>
    <tr>
      <th>270</th>
      <td>10 Things About Skid and Pump! (Friday Night Funkin' Facts)</td>
      <td>[GAMING, NEBOLIAN, DARKNEBOLIAN, DARKNEBOLAIN]</td>
      <td>270</td>
    </tr>
    <tr>
      <th>271</th>
      <td>20-Ingredient vs. 10-Ingredient vs. 2-Ingredient Chocolate Cake • Tasty</td>
      <td>[HOWTO&amp;STYLE, ALVINZHOU, ALVINZHOUCOOKING, CHOCOLATECAKE]</td>
      <td>271</td>
    </tr>
    <tr>
      <th>272</th>
      <td>21 Must Try Japanese CONVENIENCE Store Foods &amp; Drinks</td>
      <td>[ENTERTAINMENT, JAPANESECONVENIENCESTORE, JAPANESEFOOD, JAPANESESNACKS]</td>
      <td>272</td>
    </tr>
    <tr>
      <th>273</th>
      <td>@Sech, @JhayCortez  - 911 REMIX (Video Oficial)</td>
      <td>[MUSIC, SECH, PELUCHE, SECHMUSIC]</td>
      <td>273</td>
    </tr>
    <tr>
      <th>274</th>
      <td>A message from Tyler.</td>
      <td>[MUSIC, TYLERCHILDERS, LONGVIOLENTHISTORY, TYLERCHILDERSNEWALBUM]</td>
      <td>274</td>
    </tr>
    <tr>
      <th>275</th>
      <td>BTS Performs Dynamite | 2020 MTV VMAs</td>
      <td>[ENTERTAINMENT, MTV, VIDEOMUSICAWARDS, VMA]</td>
      <td>275</td>
    </tr>
    <tr>
      <th>276</th>
      <td>BTS | TIME Entertainer of the Year</td>
      <td>[NEWS&amp;POLITICS, BTS, KPOP, BTSARMY]</td>
      <td>276</td>
    </tr>
    <tr>
      <th>277</th>
      <td>BTS: Tiny Desk (Home) Concert</td>
      <td>[MUSIC, NPR, NPRMUSIC, NATIONALPUBLICRADIO]</td>
      <td>277</td>
    </tr>
    <tr>
      <th>278</th>
      <td>BUILDING a *THANOS* ROBOT ARMY in Minecraft (Insane Craft)</td>
      <td>[GAMING, None, None, None]</td>
      <td>278</td>
    </tr>
    <tr>
      <th>279</th>
      <td>Binging with Babish: Sugar Chicken from Rick &amp; Morty</td>
      <td>[ENTERTAINMENT, BINGINGWITHBABISH, BABBISH, BASICSWITHBABISH]</td>
      <td>279</td>
    </tr>
    <tr>
      <th>280</th>
      <td>Boosie Badazz - Hell's Angel</td>
      <td>[MUSIC, BOOSIE, BADAZZ, HELL'S]</td>
      <td>280</td>
    </tr>
    <tr>
      <th>281</th>
      <td>Braun Strowman uses Alexa Bliss to entice “The Fiend” Bray Wyatt: SmackDown, August 14, 2020</td>
      <td>[SPORTS, WRESTLING, SUBMISSIONWRESTLING, WWE]</td>
      <td>281</td>
    </tr>
    <tr>
      <th>282</th>
      <td>Broly's Power is Maximum in DEATH BATTLE!</td>
      <td>[ENTERTAINMENT, DEATHBATTLE, ROOSTERTEETH, RT]</td>
      <td>282</td>
    </tr>
    <tr>
      <th>283</th>
      <td>Crazy coke magic trick! (Tutorial) 🤐 #shorts</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>283</td>
    </tr>
    <tr>
      <th>284</th>
      <td>Disturbed - If I Ever Lose My Faith in You [Official Music Video]</td>
      <td>[MUSIC, DISTURBED, DAVIDDRAIMAN, DANDONEGAN]</td>
      <td>284</td>
    </tr>
    <tr>
      <th>285</th>
      <td>Dragging a shed with a track loader</td>
      <td>[AUTOS&amp;VEHICLES, None, None, None]</td>
      <td>285</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Dream ft. PmBata - Roadtrip (Official Lyric Video)</td>
      <td>[PEOPLE&amp;BLOGS, DREAMSONG, DREAMMUSIC, DREAMROADTRIP]</td>
      <td>286</td>
    </tr>
    <tr>
      <th>287</th>
      <td>Dream vs Technoblade Animation</td>
      <td>[GAMING, None, None, None]</td>
      <td>287</td>
    </tr>
    <tr>
      <th>288</th>
      <td>ETERNALS FINAL TRAILER REACTION!! (Marvel Studios' | Breakdown | Celestials)</td>
      <td>[ENTERTAINMENT, REELREJECTS, MARVELSTUDIOS’ETERNALS, FINALTRAILER]</td>
      <td>288</td>
    </tr>
    <tr>
      <th>289</th>
      <td>Elite Dangerous: Odyssey Gameplay Reveal Trailer</td>
      <td>[GAMING, ELITEDANGEROUS, ELITEFRONTIER, FRONTIER]</td>
      <td>289</td>
    </tr>
    <tr>
      <th>290</th>
      <td>First Trimester Nausea. I need advice!</td>
      <td>[ENTERTAINMENT, COLLEEN, BALLINGER, COLLEENBALLINGER]</td>
      <td>290</td>
    </tr>
    <tr>
      <th>291</th>
      <td>Friday Night Funkin but instead of Sky and Ruv there's a Different Mod every turn</td>
      <td>[GAMING, None, None, None]</td>
      <td>291</td>
    </tr>
    <tr>
      <th>292</th>
      <td>George Friendzones Dream...</td>
      <td>[PEOPLE&amp;BLOGS, MINECRAFT, CHALLENGE, MINECRAFTBUT]</td>
      <td>292</td>
    </tr>
    <tr>
      <th>293</th>
      <td>Grant - Salty 2 ft. Tiko, FaZe H1ghSky1 (Official Music Video)</td>
      <td>[GAMING, FORTNITE, FORTNITEMOBILE, FORTNITEDISSTRACK]</td>
      <td>293</td>
    </tr>
    <tr>
      <th>294</th>
      <td>Here's What It's Like To Road Trip The 702 HP Ram TRX - You'll Be Surprised By The Eye-Opening MPG!</td>
      <td>[AUTOS&amp;VEHICLES, TFLTRUCK, THEFASTLANETRUCK, RAMTRX]</td>
      <td>294</td>
    </tr>
    <tr>
      <th>295</th>
      <td>How to wake villager in Minecraft #Shorts</td>
      <td>[GAMING, None, None, None]</td>
      <td>295</td>
    </tr>
    <tr>
      <th>296</th>
      <td>I Met Corpse On The Dream SMP!</td>
      <td>[PEOPLE&amp;BLOGS, KARLJACOBS, None, None]</td>
      <td>296</td>
    </tr>
    <tr>
      <th>297</th>
      <td>I dropped a NUKE in EVERY CALL OF DUTY..</td>
      <td>[GAMING, CODMOBILE, CALLOFDUTY, NUKE]</td>
      <td>297</td>
    </tr>
    <tr>
      <th>298</th>
      <td>I managed to make a circle in minecraft</td>
      <td>[GAMING, None, None, None]</td>
      <td>298</td>
    </tr>
    <tr>
      <th>299</th>
      <td>If you don't get a stimulus check by direct deposit on Wednesday - you'll have to wait</td>
      <td>[NEWS&amp;POLITICS, VIDEO, None, None]</td>
      <td>299</td>
    </tr>
    <tr>
      <th>300</th>
      <td>Inside Nina Dobrev's 1920's Spanish-Style Home | Open Door | Architectural Digest</td>
      <td>[ENTERTAINMENT, ADCELEBRITYHOME, ADCELEBRITYHOMETOUR, ADDOBREV]</td>
      <td>300</td>
    </tr>
    <tr>
      <th>301</th>
      <td>Jennifer Lopez + Maluma - Pa' Ti + Lonely AMA's Performance 2020</td>
      <td>[MUSIC, None, None, None]</td>
      <td>301</td>
    </tr>
    <tr>
      <th>302</th>
      <td>Journalist who works in Washington D.C. talks about life after attack on Capitol</td>
      <td>[NEWS&amp;POLITICS, NEWSBRIEF, JOURNALIST, ATTACK]</td>
      <td>302</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Kane bags his first brace of the season! HIGHLIGHTS | SPURS 3-0 PACOS DE FERREIRA</td>
      <td>[SPORTS, SPURS, TOTTENHAMHOTSPUR, HARRYKANE]</td>
      <td>303</td>
    </tr>
    <tr>
      <th>304</th>
      <td>Kenny Omega vs Rich Swann Title vs Title Ring Entrances &amp; Introductions! | Rebellion 2021 Highlights</td>
      <td>[SPORTS, TNAWRESTLING, TNA, IMPACTPLUS]</td>
      <td>304</td>
    </tr>
    <tr>
      <th>305</th>
      <td>LUMPY SHINES ON OPENING DAY! | On-Season Softball Series | Game 1</td>
      <td>[SPORTS, HOMERUN, DODGERFILMS, SOFTBALL]</td>
      <td>305</td>
    </tr>
    <tr>
      <th>306</th>
      <td>LaMelo Ball Pulls Up To Mikey Williams' BIG GAME! Mikey Opens Up On San Diego! I Miss Being Home</td>
      <td>[SPORTS, MIKEYWILLIAMSLAMELO, MIKEYWILLIAMSLAMELOBALL, LAMELOBALLWATCHINGMIKEYWILLIAMS]</td>
      <td>306</td>
    </tr>
    <tr>
      <th>307</th>
      <td>Leicester City BEATS Manchester City in the FA Community Shield | ESPN FC Highlights</td>
      <td>[SPORTS, MANCHESTERCITY, MANCITY, LEICESTERCITY]</td>
      <td>307</td>
    </tr>
    <tr>
      <th>308</th>
      <td>Liam Payne - TikTok, Call of Duty, The LP Show Act 2 and a 1D Remix</td>
      <td>[PEOPLE&amp;BLOGS, LIAMPAYNE, 1D, ONEDIRECTION]</td>
      <td>308</td>
    </tr>
    <tr>
      <th>309</th>
      <td>Lizzo - Rumors feat. Cardi B [Official Video]</td>
      <td>[MUSIC, LIZZO, LIZZOMUSIC, BIGGRRRLS]</td>
      <td>309</td>
    </tr>
    <tr>
      <th>310</th>
      <td>Making Ugly Clothes Cute Challenge!! w/ Lexi Rivera</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>310</td>
    </tr>
    <tr>
      <th>311</th>
      <td>Nigel Ng (Uncle Roger) and Hersha read YOUR Comments - Part Two</td>
      <td>[COMEDY, UNCLEROGER, NIGELNG, HERSHAPATEL]</td>
      <td>311</td>
    </tr>
    <tr>
      <th>312</th>
      <td>OFFICIAL GENDER REVEAL!</td>
      <td>[HOWTO&amp;STYLE, MAKEUP, TUTORIAL, HACKS]</td>
      <td>312</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Ocho Breaks Down: Real Men Cry Too | I AM ATHLETE with Brandon Marshall, Chad Johnson &amp; More</td>
      <td>[PEOPLE&amp;BLOGS, BRANDONMARSHALL, BRANDONMARSHALLHIGHLIGHTS, BRANDONMARSHALLPODCAST]</td>
      <td>313</td>
    </tr>
    <tr>
      <th>314</th>
      <td>Panton Squad Official Music Video We Go Hard</td>
      <td>[ENTERTAINMENT, PANTONSQUAD, WEGOHARD, PANTONSQUADOFFICIALMUSICVIDEO]</td>
      <td>314</td>
    </tr>
    <tr>
      <th>315</th>
      <td>ROCKETS at LAKERS | FULL GAME HIGHLIGHTS | September 4, 2020</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>315</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Race Highlights | 2021 Dutch Grand Prix</td>
      <td>[SPORTS, F1, FORMULAONE, FORMULA1]</td>
      <td>316</td>
    </tr>
    <tr>
      <th>317</th>
      <td>Red Dead Online: Blood Money</td>
      <td>[GAMING, REDDEADONLINE, REDDEADREDEMPTION2, REDDEADONLINE:BLOODMONEYCOMINGJULY13TH]</td>
      <td>317</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Riverdale | Season 5 Trailer | The CW</td>
      <td>[ENTERTAINMENT, RIVERDALE, CW, LILIREINHART]</td>
      <td>318</td>
    </tr>
    <tr>
      <th>319</th>
      <td>Rocket League Llama-Rama Event Trailer</td>
      <td>[GAMING, LLAMA-RAMA, ROCKETLEAGUELLAMARAMA, FORTNITELLAMARAMA]</td>
      <td>319</td>
    </tr>
    <tr>
      <th>320</th>
      <td>Sanji VS Rock Lee (One Piece VS Naruto) | DEATH BATTLE!</td>
      <td>[ENTERTAINMENT, DEATHBATTLE, ROOSTERTEETH, RT]</td>
      <td>320</td>
    </tr>
    <tr>
      <th>321</th>
      <td>Sea of Thieves Season One: Official Content Update Trailer</td>
      <td>[GAMING, SEAOFTHIEVES, SEAOFTHIEVESANNIVERSARY, ANNIVERSARYUPDATE]</td>
      <td>321</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Sech, Daddy Yankee, J Balvin ft. Rosalía, Farruko - Relación Remix (Video Oficial)</td>
      <td>[MUSIC, SECH, RELACION, RELACIONREMIX]</td>
      <td>322</td>
    </tr>
    <tr>
      <th>323</th>
      <td>Shaq &amp; Chuck Have a Heated Debate Over the Kia MVP Race | NBA on TNT</td>
      <td>[SPORTS, NBAONTNT, NBA, INSIDETHENBA]</td>
      <td>323</td>
    </tr>
    <tr>
      <th>324</th>
      <td>Stranger Things 4 | Sneak Peek | Netflix</td>
      <td>[ENTERTAINMENT, CALEBMCLAUGHLIN, CHARLIEHEATON, CHIEFHOPPER]</td>
      <td>324</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Super Mario Bros. 35th Anniversary Direct</td>
      <td>[GAMING, NINTENDO, PLAY, PLAYNINTENDO]</td>
      <td>325</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Switching Houses w/ MissRemiAshten!!</td>
      <td>[ENTERTAINMENT, ALISHAMARIE, ALISHA, ALISHAMARIE]</td>
      <td>326</td>
    </tr>
    <tr>
      <th>327</th>
      <td>TASK MONSTER imposter in Among Us</td>
      <td>[GAMING, None, None, None]</td>
      <td>327</td>
    </tr>
    <tr>
      <th>328</th>
      <td>TEACHER IMPOSTER Mod in Among us</td>
      <td>[GAMING, None, None, None]</td>
      <td>328</td>
    </tr>
    <tr>
      <th>329</th>
      <td>The Adults SCP-1788 (SCP Animation)</td>
      <td>[FILM&amp;ANIMATION, THERUBBER, THERUBBER, ANIMATION]</td>
      <td>329</td>
    </tr>
    <tr>
      <th>330</th>
      <td>The Warden: Minecraft Movie | Minecraft Animation | Alex and Steve</td>
      <td>[GAMING, MINECRAFT, MINECRAFTANIMATION, BLUEMONKEY]</td>
      <td>330</td>
    </tr>
    <tr>
      <th>331</th>
      <td>This Man Can WALK FASTER Than YOU CAN RUN!</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>331</td>
    </tr>
    <tr>
      <th>332</th>
      <td>Trump administration allows Biden transition to begin</td>
      <td>[NEWS&amp;POLITICS, VIDEO, CBS, NEWS]</td>
      <td>332</td>
    </tr>
    <tr>
      <th>333</th>
      <td>Trump signs executive orders on unemployment, evictions, student loans and payroll tax</td>
      <td>[NEWS&amp;POLITICS, DEBT, STUDENTLOANS, UNEMPLOYMENTAID]</td>
      <td>333</td>
    </tr>
    <tr>
      <th>334</th>
      <td>WE HAVE A NEW GROUP MEMBER!!</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>334</td>
    </tr>
    <tr>
      <th>335</th>
      <td>WE TRADED TINY HOMES FOR 24 HOURS! (van life vs bus life)</td>
      <td>[TRAVEL&amp;EVENTS, #VANLIFE, CONVERTEDVANTOUR, KARAANDNATE]</td>
      <td>335</td>
    </tr>
    <tr>
      <th>336</th>
      <td>We're Moving To Texas!!</td>
      <td>[PEOPLE&amp;BLOGS, TEXAS, TRAVEL, None]</td>
      <td>336</td>
    </tr>
    <tr>
      <th>337</th>
      <td>Welcoming Berm Peak's Newest Resident...I'm a dad!</td>
      <td>[SPORTS, MTB, MOUNTAINBIKE, BIKEREPAIR]</td>
      <td>337</td>
    </tr>
    <tr>
      <th>338</th>
      <td>Wet N Wild x Sponge Bob Makeup… Is It Jeffree Star Approved?!</td>
      <td>[PEOPLE&amp;BLOGS, JEFFREESTAR, SPONGEBOBMAKEUP, JEFFREESTARAPPROVED]</td>
      <td>338</td>
    </tr>
    <tr>
      <th>339</th>
      <td>What if the Byzantine Empire Survived?</td>
      <td>[EDUCATION, BYZANTINEEMPIRE, WHATIFTHEBYZANTINEEMPIRESURVIVED?, ALTERNATEHISTORYHUB]</td>
      <td>339</td>
    </tr>
    <tr>
      <th>340</th>
      <td>Whoever Can Survive The Most Days In A Zombie Apocalypses In Hardcore Minecraft Wins</td>
      <td>[GAMING, FORGELABS, FORGELABS, RLCRAFT]</td>
      <td>340</td>
    </tr>
    <tr>
      <th>341</th>
      <td>ZooPhobia - Bad Luck Jack (Short)</td>
      <td>[FILM&amp;ANIMATION, VIVZIEPOP, ZOOPHOBIA, None]</td>
      <td>341</td>
    </tr>
    <tr>
      <th>342</th>
      <td>the worst part of squid game (SPOILERS)</td>
      <td>[FILM&amp;ANIMATION, None, None, None]</td>
      <td>342</td>
    </tr>
    <tr>
      <th>343</th>
      <td>【Debut PV】Hope is present【hololive English VSinger】</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>343</td>
    </tr>
    <tr>
      <th>344</th>
      <td>*NEW* FORTNITE SUMMER EVENT! (LIVE EVENT &amp; FREE REWARDS)</td>
      <td>[GAMING, FORTNITE, FORTNITESEASON7, SEASON7FORTNITE]</td>
      <td>344</td>
    </tr>
    <tr>
      <th>345</th>
      <td>2022 Mercedes-Benz EQS: an electric S-Class with over 400 miles of range</td>
      <td>[SCIENCE&amp;TECHNOLOGY, 2022MERCEDESEQS, MERCEDESEQSPRICE, MERCEDESEQSINTERIOR]</td>
      <td>345</td>
    </tr>
    <tr>
      <th>346</th>
      <td>42 Dugg - Maybach feat. Future (Official Music Video)</td>
      <td>[MUSIC, 42DUGG, 42DUGG, DUGG]</td>
      <td>346</td>
    </tr>
    <tr>
      <th>347</th>
      <td>A $1 BILLION Debt Ended 21 Years Of Greatness 😞 | #shorts</td>
      <td>[SPORTS, SHORTS, YOUTUBESHORTS, LIONELMESSI]</td>
      <td>347</td>
    </tr>
    <tr>
      <th>348</th>
      <td>AGT Winner Darci Lynne Performs Baby by Justin Bieber - America's Got Talent 2020</td>
      <td>[ENTERTAINMENT, ENTERTAINMENT, TVSERIES, HILARIOUSMUSICSONGS]</td>
      <td>348</td>
    </tr>
    <tr>
      <th>349</th>
      <td>Ana Bárbara - Angel (Video Oficial)</td>
      <td>[MUSIC, ANABARBARA, ANA, BARBARA]</td>
      <td>349</td>
    </tr>
    <tr>
      <th>350</th>
      <td>Apex Legends – Legacy Gameplay Trailer</td>
      <td>[GAMING, APEXLEGENDS, APEXLEGENDSCHARACTERS, NEWAPEXLEGEND]</td>
      <td>350</td>
    </tr>
    <tr>
      <th>351</th>
      <td>Asian Boss Is Months Away From Shutting Down</td>
      <td>[NEWS&amp;POLITICS, ASIANBOSS, ASIA, STAYCURIOUS]</td>
      <td>351</td>
    </tr>
    <tr>
      <th>352</th>
      <td>Best and Worst Dressed Met Gala 2021 (Dirty Laundry)</td>
      <td>[HOWTO&amp;STYLE, METGALA, 2021, METGALA2021]</td>
      <td>352</td>
    </tr>
    <tr>
      <th>353</th>
      <td>Billie Eilish - Therefore I Am (Live from the American Music Awards / 2020)</td>
      <td>[MUSIC, BILLIEEILISH, THEREFOREIAM, AMAS]</td>
      <td>353</td>
    </tr>
    <tr>
      <th>354</th>
      <td>BlocBoy JB - FatBoy (Intro) [Official Music Video]</td>
      <td>[MUSIC, BLOCBOY, BLOCBOYJB, BLOCBOY]</td>
      <td>354</td>
    </tr>
    <tr>
      <th>355</th>
      <td>Brawl Stars: Brawl Talk! Two New Brawlers, TONS of Skins, and a New Game mode!?</td>
      <td>[GAMING, BRAWLSTARS, MOBILEGAME, MOBILESTRATEGYGAME]</td>
      <td>355</td>
    </tr>
    <tr>
      <th>356</th>
      <td>Bruno Mars, Anderson .Paak, Silk Sonic- Leave The Door Open (Live from the iHeartRadio Music Awards)</td>
      <td>[MUSIC, BRUNO, BRUNOMARS, ANDERSONPAAK]</td>
      <td>356</td>
    </tr>
    <tr>
      <th>357</th>
      <td>Brytiago y Jay Wheeler - Desnudarte (Video Oficial)</td>
      <td>[MUSIC, QUE, BRYTIAGO, QUIERO]</td>
      <td>357</td>
    </tr>
    <tr>
      <th>358</th>
      <td>Bucket List: South Africa (Behind the Scenes)</td>
      <td>[PEOPLE&amp;BLOGS, DUDE, PERFECT, PLUS]</td>
      <td>358</td>
    </tr>
    <tr>
      <th>359</th>
      <td>Buzzfeed completely ruined my birthday</td>
      <td>[COMEDY, DANGELNO, DANGELOWALLACE, None]</td>
      <td>359</td>
    </tr>
    <tr>
      <th>360</th>
      <td>Crafting ULTIMATE CHAOTIC ARMOR in Insane Craft</td>
      <td>[GAMING, INSANECRAFT, INSANECRAFT, INSANECRAFTBIFFLE]</td>
      <td>360</td>
    </tr>
    <tr>
      <th>361</th>
      <td>DOLLAR TREE PRODUCTS YOU NEED IN YOUR LIFE! (Christmas 2020! 🎅)</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>361</td>
    </tr>
    <tr>
      <th>362</th>
      <td>Demi Lovato - Commander In Chief (Live from the Billboard Music Awards / 2020)</td>
      <td>[MUSIC, DEMILOVATO, BBMAS, BILLBOARDMUSICAWARDS]</td>
      <td>362</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Discord Server Owners Be Like...</td>
      <td>[GAMING, DISCORD, DISCORDMODS, BELUGA]</td>
      <td>363</td>
    </tr>
    <tr>
      <th>364</th>
      <td>Doja Cat - Love To Dream (Official Live Performance) | Vevo</td>
      <td>[MUSIC, DOJA, VEVO, RCA]</td>
      <td>364</td>
    </tr>
    <tr>
      <th>365</th>
      <td>EMPTY HOUSE TOUR!! | Louie's Life</td>
      <td>[ENTERTAINMENT, EMPTY, HOUSE, TOUR]</td>
      <td>365</td>
    </tr>
    <tr>
      <th>366</th>
      <td>El Mimoso - Basta Ya  - (Official Music Video) - (Popurri) - 2021</td>
      <td>[MUSIC, ELMIMOSOMIX, ELMIMOSOEXITOS, ELMIMOSOCOVERS]</td>
      <td>366</td>
    </tr>
    <tr>
      <th>367</th>
      <td>England v Pakistan - Highlights | Magic Mahmood Takes 4! | 1st Men’s Royal London ODI 2021</td>
      <td>[SPORTS, CRICKETVIDEOS, HIGHLIGHTS, CRICKET]</td>
      <td>367</td>
    </tr>
    <tr>
      <th>368</th>
      <td>Epic Seafood Chopped! 2HYPE</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>368</td>
    </tr>
    <tr>
      <th>369</th>
      <td>FAZE CLAN PLAYS SQUID GAME</td>
      <td>[GAMING, FAZE, FAZECLAN, FAZECLAN]</td>
      <td>369</td>
    </tr>
    <tr>
      <th>370</th>
      <td>Final Jeopardy! 10/11/21 Plus Exclusive Overheard On Set Clip | JEOPARDY!</td>
      <td>[ENTERTAINMENT, DOUBLEJEOPARDY!, EXCLUSIVEJEOPARDY, JEOPARDYQUESTIONS]</td>
      <td>370</td>
    </tr>
    <tr>
      <th>371</th>
      <td>Fredo Bang - Don't Stop Believing (Official Video)</td>
      <td>[MUSIC, FREDOBANG, LILDURK, DURKIO]</td>
      <td>371</td>
    </tr>
    <tr>
      <th>372</th>
      <td>GAME  (Full Video)  Shooter Kahlon | Sidhu Moose Wala | Hunny PK Films | Gold Media | 5911 Records</td>
      <td>[MUSIC, SIDHUMOOSEWALA, GAMEOFFICIALVIDEO, SHOOTERKAHLON]</td>
      <td>372</td>
    </tr>
    <tr>
      <th>373</th>
      <td>GODZILLA VS KONG - TRAILER REACTION!! (It's Finally Here! | MechaGodzilla?!)</td>
      <td>[ENTERTAINMENT, GODZILLAVSKONGTRAILERREACTION, GODZILLAVS.KONG–OFFICIALTRAILER, GODZILLAVSKONGTRAILERREACTION]</td>
      <td>373</td>
    </tr>
    <tr>
      <th>374</th>
      <td>Gary Neville reacts to Cristiano Ronaldo re-signing for Manchester United</td>
      <td>[SPORTS, EPLOTHER1920, FOOTBALL, FOOTBALLLEAGUE]</td>
      <td>374</td>
    </tr>
    <tr>
      <th>375</th>
      <td>Getting Glam With MEGAN THEE STALLION! | NikkieTutorials</td>
      <td>[ENTERTAINMENT, MEGANTHEESTALLION, MEGANTHESTALLION, MEGHANTHEESTALLION]</td>
      <td>375</td>
    </tr>
    <tr>
      <th>376</th>
      <td>HOW WE PICKED OUR BABIES' NAMES | BABY GIRL COMES HOME FROM THE NICU</td>
      <td>[ENTERTAINMENT, ARIEANDLAUREN, LAURENANDARIE, THEBACHELOR]</td>
      <td>376</td>
    </tr>
    <tr>
      <th>377</th>
      <td>HUGE Rampart Update! Apex Legends Evolution Event Patch Notes &amp; Trailer Reaction!</td>
      <td>[GAMING, APEXLEGENDS, APEX, PLAYAPEX]</td>
      <td>377</td>
    </tr>
    <tr>
      <th>378</th>
      <td>Hitting Rock Bottom</td>
      <td>[PEOPLE&amp;BLOGS, ROMAN, ATWOOD, THEATWOODS]</td>
      <td>378</td>
    </tr>
    <tr>
      <th>379</th>
      <td>Home Sweet Home Alone | Official Trailer | Disney+</td>
      <td>[FILM&amp;ANIMATION, TRAILER, HOMEALONE, OFFICIALTRAILER]</td>
      <td>379</td>
    </tr>
    <tr>
      <th>380</th>
      <td>How Russell seriously upset Mercedes in huge Bottas F1 crash</td>
      <td>[SPORTS, F1, FORMULA1, FORMULAONE]</td>
      <td>380</td>
    </tr>
    <tr>
      <th>381</th>
      <td>How many robots does it take to run a grocery store?</td>
      <td>[EDUCATION, TOMSCOTT, TOMSCOTT, None]</td>
      <td>381</td>
    </tr>
    <tr>
      <th>382</th>
      <td>How to Make BOMB AF SPICY NOODLES!! | Louie's Life</td>
      <td>[ENTERTAINMENT, HOWTO, HOWTOMAKE, HOWTOCOOK]</td>
      <td>382</td>
    </tr>
    <tr>
      <th>383</th>
      <td>I DIY'd MY SUMMER WARDROBE</td>
      <td>[ENTERTAINMENT, FUNCTION, OF, BEAUTY]</td>
      <td>383</td>
    </tr>
    <tr>
      <th>384</th>
      <td>I Made Squid Game, But it's a Multiplayer Game</td>
      <td>[ENTERTAINMENT, UNITY, UNITYGAMEDEV, UNITYGAMEDEVLOG]</td>
      <td>384</td>
    </tr>
    <tr>
      <th>385</th>
      <td>I Made The Best VRChat Avatar</td>
      <td>[GAMING, None, None, None]</td>
      <td>385</td>
    </tr>
    <tr>
      <th>386</th>
      <td>I've never seen my best friend this angry..</td>
      <td>[ENTERTAINMENT, FAZERUG, RUG, RUGFAZE]</td>
      <td>386</td>
    </tr>
    <tr>
      <th>387</th>
      <td>IS FOOD REALLY THAT BAD IN PRISON?</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>387</td>
    </tr>
    <tr>
      <th>388</th>
      <td>If There Was a Button That Turned You Instantly Productive.</td>
      <td>[COMEDY, PRODUCTIVITY, FOCUS, HARDWORK]</td>
      <td>388</td>
    </tr>
    <tr>
      <th>389</th>
      <td>In The Heights - First 8 Minutes</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>389</td>
    </tr>
    <tr>
      <th>390</th>
      <td>Introducing Apple Watch Series 6 — It Already Does That</td>
      <td>[SCIENCE&amp;TECHNOLOGY, APPLEWATCHSERIES6, APPLEWATCH, APPLE]</td>
      <td>390</td>
    </tr>
    <tr>
      <th>391</th>
      <td>KSI WAS MY ASSISTANT FOR 24 HOURS!</td>
      <td>[GAMING, VIKKSTAR, VIKKSTAR123, VIKKSTAR123HD]</td>
      <td>391</td>
    </tr>
    <tr>
      <th>392</th>
      <td>King Von ft. Fivio Foreign - I Am What I Am (Official Video)</td>
      <td>[MUSIC, KINGVON, KINGVON2018, KINGVONOFFICIAL]</td>
      <td>392</td>
    </tr>
    <tr>
      <th>393</th>
      <td>La Ross Maria x La Perversa - Klk El Dice ( Video Oficial )</td>
      <td>[MUSIC, None, None, None]</td>
      <td>393</td>
    </tr>
    <tr>
      <th>394</th>
      <td>Los Dos Carnales - El Borracho (Video Oficial)</td>
      <td>[MUSIC, LOSDOSCARNALES, LOSDOSCARNALESELBORRACHO, ELBORRACHO]</td>
      <td>394</td>
    </tr>
    <tr>
      <th>395</th>
      <td>MAKING MY QUEEN’S 26th BIRTHDAY SPECIAL VLOG 💕 (SURPRISE DREAM CAR) * BTS BUTTERFLY TOUR‼️</td>
      <td>[ENTERTAINMENT, QUEENNAIJA, MEDICINE, QUEEN]</td>
      <td>395</td>
    </tr>
    <tr>
      <th>396</th>
      <td>Megan Thee Stallion | Nike “New Hotties</td>
      <td>[MUSIC, MEGANTHEESTALLION, RAP, HOUSTON]</td>
      <td>396</td>
    </tr>
    <tr>
      <th>397</th>
      <td>Mid-Season Sneak Peek | Marvel Studios' Loki | Disney+</td>
      <td>[ENTERTAINMENT, MARVEL, COMICS, LOKI]</td>
      <td>397</td>
    </tr>
    <tr>
      <th>398</th>
      <td>Minecraft but there's Custom Boats</td>
      <td>[GAMING, MINECRAFT, MINECRAFTBUT, NEWMINECRAFT]</td>
      <td>398</td>
    </tr>
    <tr>
      <th>399</th>
      <td>Minecraft but you can Eat Mobs</td>
      <td>[GAMING, MINECRAFT, MINECRAFTBUT, NEWMINECRAFT]</td>
      <td>399</td>
    </tr>
    <tr>
      <th>400</th>
      <td>Money Man - 24 (Audio) (feat. Lil Baby)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>400</td>
    </tr>
    <tr>
      <th>401</th>
      <td>Music Producer Reacts to K/DA - THE BADDEST ft. (G)I-DLE, Bea Miller, Wolftyla</td>
      <td>[MUSIC, K/DA, KDA, THEBADDEST]</td>
      <td>401</td>
    </tr>
    <tr>
      <th>402</th>
      <td>My friends made me build kitchenware out of public restroom parts</td>
      <td>[SCIENCE&amp;TECHNOLOGY, None, None, None]</td>
      <td>402</td>
    </tr>
    <tr>
      <th>403</th>
      <td>PARENTS NIGHT OUT! *Wildin for Cristianblends Bday*</td>
      <td>[PEOPLE&amp;BLOGS, BENNYSOLIVEN, ALONDRADESSY, ELSYGUEVARA]</td>
      <td>403</td>
    </tr>
    <tr>
      <th>404</th>
      <td>Retirement | JJ Redick</td>
      <td>[SPORTS, JJREDICK, JJREDICKYOUTUBE, JJREDICKRETIREMENT]</td>
      <td>404</td>
    </tr>
    <tr>
      <th>405</th>
      <td>Road to Season 2 Trailer | The Witcher</td>
      <td>[ENTERTAINMENT, ANDRZEJSAPKOWSKI, ANYACHALOTRA, BTS]</td>
      <td>405</td>
    </tr>
    <tr>
      <th>406</th>
      <td>SOMETHING BIG IS COMING!!!!</td>
      <td>[ENTERTAINMENT, COLIN, FURZE, COMINGSOON]</td>
      <td>406</td>
    </tr>
    <tr>
      <th>407</th>
      <td>STRAY KIDS Decide Which Band Member is the Best Singer, Cutest, Funniest, and More | Superlatives</td>
      <td>[HOWTO&amp;STYLE, STRAYKIDS, STRAYKIDSREACTION, STRAYKIDSBTS]</td>
      <td>407</td>
    </tr>
    <tr>
      <th>408</th>
      <td>Should I Call This Girl’s Number (tiktok) #shorts</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>408</td>
    </tr>
    <tr>
      <th>409</th>
      <td>Should you upgrade to the iPhone 12 and iPhone 12 Pro?</td>
      <td>[SCIENCE&amp;TECHNOLOGY, IPHONE12, IPHONE122020, IPHONE12COST]</td>
      <td>409</td>
    </tr>
    <tr>
      <th>410</th>
      <td>Skyrim, but random enemies spawn every time I'm hit</td>
      <td>[GAMING, DOUGDOUG, DOUGDOUGYOUTUBECHANNEL, CHANNELYOUTUBEDOUGDOUG]</td>
      <td>410</td>
    </tr>
    <tr>
      <th>411</th>
      <td>Spider-Man No Way Home Trailer REACTION! Green Goblin CONFIRMED!</td>
      <td>[ENTERTAINMENT, NEWROCKSTARS, NEWROCKSTARSYOUTUBE, YOUTUBENEWROCKSTARS]</td>
      <td>411</td>
    </tr>
    <tr>
      <th>412</th>
      <td>Spies in Disguise - Death of a Studio</td>
      <td>[ENTERTAINMENT, NOSTALGIA, NETFLIX, WILLSMITH]</td>
      <td>412</td>
    </tr>
    <tr>
      <th>413</th>
      <td>Super Nintendo World Direct 12.18.2020</td>
      <td>[GAMING, NINTENDO, PLAY, PLAYNINTENDO]</td>
      <td>413</td>
    </tr>
    <tr>
      <th>414</th>
      <td>TO YOUR ETERNITY: A JOURNEY OF PAIN, SADNESS, AND, SORROW</td>
      <td>[FILM&amp;ANIMATION, None, None, None]</td>
      <td>414</td>
    </tr>
    <tr>
      <th>415</th>
      <td>The Dixie D'Amelio Show with Jaden Hossler</td>
      <td>[PEOPLE&amp;BLOGS, DIXIEDAMELIO, DIXIED'AMELIO, DIXIE]</td>
      <td>415</td>
    </tr>
    <tr>
      <th>416</th>
      <td>The IT Guy Exposes Everyone's Secrets - The Office US</td>
      <td>[ENTERTAINMENT, THEOFFICE, THEOFFICEFULLEPISODES, RAINNWILSON]</td>
      <td>416</td>
    </tr>
    <tr>
      <th>417</th>
      <td>The Sims 4 Star Wars: Journey to Batuu | Official Reveal Trailer</td>
      <td>[GAMING, THESIMS4, THESIMS4TRAILER, THESIMS4GAMEPLAY]</td>
      <td>417</td>
    </tr>
    <tr>
      <th>418</th>
      <td>The Universe is Hostile to Computers</td>
      <td>[EDUCATION, VERITASIUM, SCIENCE, PHYSICS]</td>
      <td>418</td>
    </tr>
    <tr>
      <th>419</th>
      <td>The View Co-Hosts React to Derek Chauvin Verdict| The View</td>
      <td>[ENTERTAINMENT, THEVIEW, WHOOPIGOLDBERG, JOYBEHAR]</td>
      <td>419</td>
    </tr>
    <tr>
      <th>420</th>
      <td>The most important day of my life</td>
      <td>[PEOPLE&amp;BLOGS, JAKEPAUL, None, None]</td>
      <td>420</td>
    </tr>
    <tr>
      <th>421</th>
      <td>Tokischa x Secreto el Famoso Biberon - No Me Importa (Video Oficial)</td>
      <td>[MUSIC, TOKISCHA, SECRETO, TOKICHA]</td>
      <td>421</td>
    </tr>
    <tr>
      <th>422</th>
      <td>Tyrone Magnus Interviews ZACK SNYDER!  #RestoreTheSnyderVerse</td>
      <td>[COMEDY, ZACKSNYDER, INTERVIEW, TYRONEMAGNUS]</td>
      <td>422</td>
    </tr>
    <tr>
      <th>423</th>
      <td>UFC 256 Embedded: Vlog Series - Episode 2</td>
      <td>[SPORTS, UFC, ULTIMATE, FIGHTING]</td>
      <td>423</td>
    </tr>
    <tr>
      <th>424</th>
      <td>We Are Moving In Together !!!</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>424</td>
    </tr>
    <tr>
      <th>425</th>
      <td>We Need Some Time Apart</td>
      <td>[TRAVEL&amp;EVENTS, None, None, None]</td>
      <td>425</td>
    </tr>
    <tr>
      <th>426</th>
      <td>We Proved Royal Experts Lie About Harry and Meghan</td>
      <td>[COMEDY, JOSHPIETERS, JOSHUAPIETERS, None]</td>
      <td>426</td>
    </tr>
    <tr>
      <th>427</th>
      <td>What are the black flags at the US Capitol and why are there so many buses around the Congress?</td>
      <td>[NEWS&amp;POLITICS, CAPITOLBUSES, CONGRESSBUSES, CAPITOLFLAG]</td>
      <td>427</td>
    </tr>
    <tr>
      <th>428</th>
      <td>Where Gaming Begins: Ep. 2 | AMD Radeon™ RX 6000 Series Graphics Cards</td>
      <td>[SCIENCE&amp;TECHNOLOGY, AMD, ADVANCEDMICRODEVICES, RADEON]</td>
      <td>428</td>
    </tr>
    <tr>
      <th>429</th>
      <td>Why Don't We - Lotus Inn [Official Audio]</td>
      <td>[MUSIC, WHYDONTWE, WHYDON'TWE, FALLIN']</td>
      <td>429</td>
    </tr>
    <tr>
      <th>430</th>
      <td>Would You Sit In Snakes For $10,000?</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>430</td>
    </tr>
    <tr>
      <th>431</th>
      <td>[Vinesauce] Vinny reacts to the Cast of the Mario Movie</td>
      <td>[GAMING, VINNY, VINESAUCE, FULLSAUCE]</td>
      <td>431</td>
    </tr>
    <tr>
      <th>432</th>
      <td>attempting to cook for the first time</td>
      <td>[ENTERTAINMENT, ENTERTAINMENT, VLOG, FUNNY]</td>
      <td>432</td>
    </tr>
    <tr>
      <th>433</th>
      <td>iOS 15 - 19 Settings You NEED to Change Immediately!</td>
      <td>[SCIENCE&amp;TECHNOLOGY, IOS15, IOS15SETTINGS, IOS15TRICKS]</td>
      <td>433</td>
    </tr>
    <tr>
      <th>434</th>
      <td>my mom told me to go play outside.. So I Made My DREAM TREEHOUSE GAMING SETUP!!</td>
      <td>[GAMING, FORTNITE, GAMING, TWITCH]</td>
      <td>434</td>
    </tr>
    <tr>
      <th>435</th>
      <td>the guys from fortnite</td>
      <td>[ENTERTAINMENT, SHORTS, LAZARBEAM, LAZAR]</td>
      <td>435</td>
    </tr>
    <tr>
      <th>436</th>
      <td>Ángela Aguilar - En Realidad (Video Oficial)</td>
      <td>[MUSIC, ANGELAAGUILAR, ANGELAAGUILAR2021, NUEVOANGELAAGUILAR]</td>
      <td>436</td>
    </tr>
    <tr>
      <th>437</th>
      <td>ØMI - You (Prod. SUGA of BTS) -Official Music Video-</td>
      <td>[MUSIC, OMI, HIROOMITOSAKA, 登坂広臣]</td>
      <td>437</td>
    </tr>
    <tr>
      <th>438</th>
      <td>#4 CLIPPERS at #2 SUNS | FULL GAME HIGHLIGHTS | June 22, 2021</td>
      <td>[SPORTS, BASKETBALL, GLEAGUE, NBA]</td>
      <td>438</td>
    </tr>
    <tr>
      <th>439</th>
      <td>*NEW* SEASON 6 BATTLEPASS In Fortnite (Lara Croft + More)</td>
      <td>[GAMING, LACHLANLACHYFORTNITEBATTLEROYALEPUBGBATTLEGROUNDS, FORTNITEBATTLEROYALE, FORTNITE]</td>
      <td>439</td>
    </tr>
    <tr>
      <th>440</th>
      <td>2003 Iraq War (2/2) | Animated History</td>
      <td>[EDUCATION, None, None, None]</td>
      <td>440</td>
    </tr>
    <tr>
      <th>441</th>
      <td>Apple Watch Series 6, Apple Watch SE and new iPad Air!</td>
      <td>[PEOPLE&amp;BLOGS, IJUSTINE, None, None]</td>
      <td>441</td>
    </tr>
    <tr>
      <th>442</th>
      <td>Apple's Massive Product Launch - New 10.8 iPad Air, iPhone 12, Apple Watch Series 6, and More!</td>
      <td>[SCIENCE&amp;TECHNOLOGY, APPLESEPTEMBEREVENT, APPLEEVENT2020, APPLEIPHONEEVENT2020]</td>
      <td>442</td>
    </tr>
    <tr>
      <th>443</th>
      <td>Arizona's MyKayla Skinner wins silver!</td>
      <td>[NEWS&amp;POLITICS, OLYMPICS, OTT, SPORTS]</td>
      <td>443</td>
    </tr>
    <tr>
      <th>444</th>
      <td>Assassin's Creed Valhalla Review</td>
      <td>[GAMING, IGN, XBOX, REVIEW]</td>
      <td>444</td>
    </tr>
    <tr>
      <th>445</th>
      <td>Asteroid Impact: What Are Our Chances?</td>
      <td>[EDUCATION, VERITASIUM, SCIENCE, PHYSICS]</td>
      <td>445</td>
    </tr>
    <tr>
      <th>446</th>
      <td>Aventura, Bad Bunny - Volví (Video Oficial)</td>
      <td>[MUSIC, BADBUNNY, BADBUNNYAVENTURA, BADBUNNYVOLVI]</td>
      <td>446</td>
    </tr>
    <tr>
      <th>447</th>
      <td>Bears vs. Falcons Week 3 Highlights | NFL 2020</td>
      <td>[SPORTS, NFL, FOOTBALL, OFFENSE]</td>
      <td>447</td>
    </tr>
    <tr>
      <th>448</th>
      <td>Big Scarr Covers Gucci Mane's Hit Song Big Boy Diamonds I 17 Bars</td>
      <td>[MUSIC, AUDIOMACK, AUDIOMACK, TRAPSYMPHONY]</td>
      <td>448</td>
    </tr>
    <tr>
      <th>449</th>
      <td>Big30 ft. DeeMula &amp; Pooh Shiesty - Neighborhood Heroes (Official Video)</td>
      <td>[MUSIC, #CGE, #BREADGANG, #NLESS]</td>
      <td>449</td>
    </tr>
    <tr>
      <th>450</th>
      <td>Billie Eilish - Therefore I Am (Live From The ARIAS)</td>
      <td>[MUSIC, BILLIE, EILISH, THEREFORE]</td>
      <td>450</td>
    </tr>
    <tr>
      <th>451</th>
      <td>Brawl Stars Animation: Welcome to the Gift Shop!</td>
      <td>[GAMING, BRAWLSTARS, MOBILEGAME, MOBILESTRATEGYGAME]</td>
      <td>451</td>
    </tr>
    <tr>
      <th>452</th>
      <td>Candyman</td>
      <td>[COMEDY, DOMICS, ANIMATION, None]</td>
      <td>452</td>
    </tr>
    <tr>
      <th>453</th>
      <td>Candyman - Official Trailer 2</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>453</td>
    </tr>
    <tr>
      <th>454</th>
      <td>Canelo's HYPED Reaction To Beating Billy Joe Saunders, Calls Out Caleb Plant</td>
      <td>[SPORTS, DAZN, BOXING, HIGHLIGHTS]</td>
      <td>454</td>
    </tr>
    <tr>
      <th>455</th>
      <td>DJ Khaled - EVERY CHANCE I GET (Official Audio) ft. Lil Baby, Lil Durk</td>
      <td>[MUSIC, FATHEROFASHAD, ASHAD, I'MTHEONE]</td>
      <td>455</td>
    </tr>
    <tr>
      <th>456</th>
      <td>Doctor Reacts to Severe Eczema Problems #shorts #eczema</td>
      <td>[EDUCATION, None, None, None]</td>
      <td>456</td>
    </tr>
    <tr>
      <th>457</th>
      <td>Dodge is FINALLY going electric, but is it too late?</td>
      <td>[AUTOS&amp;VEHICLES, None, None, None]</td>
      <td>457</td>
    </tr>
    <tr>
      <th>458</th>
      <td>Draymond and KD Reveal What Really Happened with Warriors Fallout | FULL INTERVIEW (Chips)</td>
      <td>[SPORTS, BLEACHERREPORT, BR, NBA]</td>
      <td>458</td>
    </tr>
    <tr>
      <th>459</th>
      <td>EVIL iPhone IMPOSTER in Among Us</td>
      <td>[GAMING, None, None, None]</td>
      <td>459</td>
    </tr>
    <tr>
      <th>460</th>
      <td>Escaping Minecraft's Most Perfect Prison (gaia's vault v3) ft. SeenSven</td>
      <td>[GAMING, MINECRAFT, DREAM, DREAMSMP]</td>
      <td>460</td>
    </tr>
    <tr>
      <th>461</th>
      <td>FULL EPISODE | Rick and Morty Season 5 Premiere: Mort Dinner Rick Andre | adult swim</td>
      <td>[ENTERTAINMENT, ADULTSWIM, ANIMATION, ADULTANIMATION]</td>
      <td>461</td>
    </tr>
    <tr>
      <th>462</th>
      <td>Friday Night Funkin' Mod Characters Reacts | Part 9 | Moonlight Cactus |</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>462</td>
    </tr>
    <tr>
      <th>463</th>
      <td>Friday Night Funkin' but Different Characters Sings Ugh</td>
      <td>[GAMING, AMONGUSDRIP, AMOGUS, AMONGUSINFRIDAYNIGHTFUNKIN]</td>
      <td>463</td>
    </tr>
    <tr>
      <th>464</th>
      <td>G Herbo - Stand the Rain (Mad Max) (Official Music Video)</td>
      <td>[MUSIC, HERBO, STAND, THE]</td>
      <td>464</td>
    </tr>
    <tr>
      <th>465</th>
      <td>HEAT at LAKERS | FULL GAME HIGHLIGHTS | October 9, 2020</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>465</td>
    </tr>
    <tr>
      <th>466</th>
      <td>HOW WE MET -- GF REVEAL</td>
      <td>[COMEDY, RACHEL, BALLINGER, NOVAQUA]</td>
      <td>466</td>
    </tr>
    <tr>
      <th>467</th>
      <td>Hidilyn Diaz wins the Philippines' first-ever Olympic gold medal | Tokyo Olympics | NBC Sports</td>
      <td>[SPORTS, OLYMPICS, NBC, NBCSPORTS]</td>
      <td>467</td>
    </tr>
    <tr>
      <th>468</th>
      <td>Highlights | Newcastle 1-4 Manchester United | Rampant Reds come from behind to claim big win</td>
      <td>[SPORTS, MANCHESTERUNITED, MUFC, MANUTD]</td>
      <td>468</td>
    </tr>
    <tr>
      <th>469</th>
      <td>I Built a MANSION out of MELONS in Minecraft Hardcore (#41)</td>
      <td>[GAMING, IBUILTAMANSIONOUTOFMELONSINMINECRAFTHARDCORE, IBUILTAMANSIONINMINECRAFT, IBUILTAGIANTMANSIONINMINECRAFT]</td>
      <td>469</td>
    </tr>
    <tr>
      <th>470</th>
      <td>I Haven't Been Honest About My Injury.. Here's THE TRUTH</td>
      <td>[ENTERTAINMENT, PROFESSORINJURY, PROFESSORACHILLES, PROFESSOR1V1]</td>
      <td>470</td>
    </tr>
    <tr>
      <th>471</th>
      <td>I Let Ali-A Control My Fortnite Game... *regret*</td>
      <td>[GAMING, FORTNITE, CHALLENGE, FUNNY]</td>
      <td>471</td>
    </tr>
    <tr>
      <th>472</th>
      <td>I Stole my Friends Among Us Statue</td>
      <td>[GAMING, REKRAP2, REKRAP, REK]</td>
      <td>472</td>
    </tr>
    <tr>
      <th>473</th>
      <td>I bought every ad I saw on instagram for a week</td>
      <td>[COMEDY, DREWGOODEN, COMEDY, COMMENTARY]</td>
      <td>473</td>
    </tr>
    <tr>
      <th>474</th>
      <td>I'm not a good cook (but I'm trying)</td>
      <td>[FILM&amp;ANIMATION, ILLYMATION, ILLYMATIONS, ILLYANIMATION]</td>
      <td>474</td>
    </tr>
    <tr>
      <th>475</th>
      <td>If You Don't Love It When Your Coach Does This Something Is Wrong</td>
      <td>[PEOPLE&amp;BLOGS, IFYOUDON'TLOVEITWHENYOURCOACHDOESTHISSOMETHINGISWRONG, #SHORTS, None]</td>
      <td>475</td>
    </tr>
    <tr>
      <th>476</th>
      <td>J. Cole - i n t e r l u d e (Official Audio)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>476</td>
    </tr>
    <tr>
      <th>477</th>
      <td>Juice WRLD ft. Clever &amp; Post Malone - Life's A Mess II (Official Visualizer)</td>
      <td>[MUSIC, CLEVER, WHOISCLEVER, None]</td>
      <td>477</td>
    </tr>
    <tr>
      <th>478</th>
      <td>Kevin Durant Joins Inside the NBA After His First Game With the Brooklyn Nets | NBA on TNT</td>
      <td>[SPORTS, NBAONTNT, NBA, INSIDETHENBA]</td>
      <td>478</td>
    </tr>
    <tr>
      <th>479</th>
      <td>Knock the Block Challenge!! *Don't FALL off the Tower!*</td>
      <td>[COMEDY, TEAMEDGE, TEAMEDGECHALLENGE, TEAMEDGE]</td>
      <td>479</td>
    </tr>
    <tr>
      <th>480</th>
      <td>LAKERS at HAWKS | FULL GAME HIGHLIGHTS | February 1, 2021</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>480</td>
    </tr>
    <tr>
      <th>481</th>
      <td>LISA - 'LALISA' Live Performance Stage | OUTNOW Unlimited 210914</td>
      <td>[MUSIC, MUPLY, 네이버나우, OUTNOWUNLIMITED]</td>
      <td>481</td>
    </tr>
    <tr>
      <th>482</th>
      <td>LISA - 'LALISA' M/V</td>
      <td>[MUSIC, YGENTERTAINMENT, YG, 와이지]</td>
      <td>482</td>
    </tr>
    <tr>
      <th>483</th>
      <td>Lil Tjay &amp; 6LACK - Calling My Phone (Lyric Video)</td>
      <td>[MUSIC, LILTJAY, STEADYCALLINGMYPHONE, CALLINGMYPHONETIKTOK]</td>
      <td>483</td>
    </tr>
    <tr>
      <th>484</th>
      <td>MORTAL KOMBAT Official Trailer (2021)</td>
      <td>[ENTERTAINMENT, MORTALKOMBAT, MORTALKOMBAT, MORTALKOMBATMOVIE]</td>
      <td>484</td>
    </tr>
    <tr>
      <th>485</th>
      <td>Mace Windu's Return in The Mandalorian Season 2 - Star Wars Theory</td>
      <td>[ENTERTAINMENT, STARWARSEXPLAINED, STARWARSTHEORY, DARTHVADERLIGHTSABER]</td>
      <td>485</td>
    </tr>
    <tr>
      <th>486</th>
      <td>Man fatally shot while walking dog in NYC</td>
      <td>[NEWS&amp;POLITICS, BAYRIDGESHOOTING, MANWALKINGDOGSHOT, FATALSHOOTING]</td>
      <td>486</td>
    </tr>
    <tr>
      <th>487</th>
      <td>Messi nets two goals for Argentina in 4-1 win over Bolivia | 2021 Copa America Highlights</td>
      <td>[SPORTS, FOX, FOXSPORTS, SOCCER]</td>
      <td>487</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Microsoft Windows 11 event in 7 minutes: Android Apps, New Start Menu, Free Upgrade</td>
      <td>[SCIENCE&amp;TECHNOLOGY, WINDOWS11, WINDOWS, ANDROID]</td>
      <td>488</td>
    </tr>
    <tr>
      <th>489</th>
      <td>Minecraft But Slimes Beat the game for you...</td>
      <td>[GAMING, MINECRAFT, NEWITEMSINMINECRAFT, NEWWEAPONS]</td>
      <td>489</td>
    </tr>
    <tr>
      <th>490</th>
      <td>Minecraft, But If You Think Any Item, You Get It...</td>
      <td>[GAMING, MINECRAFT, MINECRAFTBUT, MCBUT]</td>
      <td>490</td>
    </tr>
    <tr>
      <th>491</th>
      <td>Minecraft: Working Trash Bin! #Shorts</td>
      <td>[GAMING, MINECRAFT, #SHORT, #SHORTS]</td>
      <td>491</td>
    </tr>
    <tr>
      <th>492</th>
      <td>Mini Crewmate Kills Little Nightmares Characters | Among Us</td>
      <td>[FILM&amp;ANIMATION, AMONGUS, AMONGUSANIMATION, AMONGUSKILLANIMATIONS]</td>
      <td>492</td>
    </tr>
    <tr>
      <th>493</th>
      <td>NASA's SpaceX Crew-1 Mission Splashes Down</td>
      <td>[SCIENCE&amp;TECHNOLOGY, None, None, None]</td>
      <td>493</td>
    </tr>
    <tr>
      <th>494</th>
      <td>OVER POWERED POINT GUARD BUILD IN NBA 2K22! BEST BUILD FOR SHOOTING &amp; DRIBBLING</td>
      <td>[GAMING, CHEESEAHOLIC, NBA, NBA2K22]</td>
      <td>494</td>
    </tr>
    <tr>
      <th>495</th>
      <td>Overwatch Archives 2021 | Overwatch Seasonal Event</td>
      <td>[GAMING, OVERWATCHEVENT, ARCHIVES, OVERWATCHEVENT]</td>
      <td>495</td>
    </tr>
    <tr>
      <th>496</th>
      <td>PANTRY TOUR!!!!!</td>
      <td>[FILM&amp;ANIMATION, 8-7-20, None, None]</td>
      <td>496</td>
    </tr>
    <tr>
      <th>497</th>
      <td>PIPER ROCKELLE vs ELLIANA WALMSLEY Viral TikTok Challenge</td>
      <td>[ENTERTAINMENT, JORDANMATTER, DANCEPHOTOGRAPHY, CHALLENGE]</td>
      <td>497</td>
    </tr>
    <tr>
      <th>498</th>
      <td>POP SMOKE X JAY GWUAPO - BLACK MASK (OFFICIAL VIDEO)</td>
      <td>[MUSIC, POPSMOKE, POPSMOKEJAYGWUAPO, JAYGWUAPOBLACKMASK]</td>
      <td>498</td>
    </tr>
    <tr>
      <th>499</th>
      <td>POV: You and your minecraft friend argue about who has more stuff</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>499</td>
    </tr>
    <tr>
      <th>500</th>
      <td>PS5 - 17 Things You Need To Know Before You Buy | Pre-Order</td>
      <td>[GAMING, EVERYTHINGYOUNEEDTOKNOWABOUTPS5, PS5PRE-ORDER, PS5DETAILS]</td>
      <td>500</td>
    </tr>
    <tr>
      <th>501</th>
      <td>Peyton Manning Full Hall of Fame Speech | 2021 Pro Football Hall of Fame | NFL</td>
      <td>[SPORTS, None, None, None]</td>
      <td>501</td>
    </tr>
    <tr>
      <th>502</th>
      <td>Quando Rondo - I Thought</td>
      <td>[MUSIC, QUANDORONDO, ATLANTIC, RECORDS]</td>
      <td>502</td>
    </tr>
    <tr>
      <th>503</th>
      <td>Resumen y goles | México 3-2 Corea del Sur | Amistoso 2020 | TUDN</td>
      <td>[SPORTS, TELEVISA, UNIVISION, TUDN]</td>
      <td>503</td>
    </tr>
    <tr>
      <th>504</th>
      <td>Rich Mom Vs. Broke Mom</td>
      <td>[ENTERTAINMENT, YOUTWOTV, YOUTWO, SINCERELYJAZ]</td>
      <td>504</td>
    </tr>
    <tr>
      <th>505</th>
      <td>Riding out to a Great RC Crawling Spot!</td>
      <td>[HOWTO&amp;STYLE, DOITYOURSELF, DIY, TOOLS]</td>
      <td>505</td>
    </tr>
    <tr>
      <th>506</th>
      <td>Rod Wave's 2020 XXL Freshman Freestyle</td>
      <td>[MUSIC, RODWAVE, RAPPER, RAP]</td>
      <td>506</td>
    </tr>
    <tr>
      <th>507</th>
      <td>Shang-Chi and the Legend of the Ten Rings - Movie Review</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>507</td>
    </tr>
    <tr>
      <th>508</th>
      <td>Simps</td>
      <td>[FILM&amp;ANIMATION, SWOOZIE, ADANDE, ANIMATION]</td>
      <td>508</td>
    </tr>
    <tr>
      <th>509</th>
      <td>Sizzle | The Bad Batch | Disney+</td>
      <td>[ENTERTAINMENT, STARWARS, BADBATCH, DISNEY+]</td>
      <td>509</td>
    </tr>
    <tr>
      <th>510</th>
      <td>Sony PLAYSTATION 5 Digital Edition UNBOXING &amp; PS5 GIVEAWAY</td>
      <td>[SCIENCE&amp;TECHNOLOGY, None, None, None]</td>
      <td>510</td>
    </tr>
    <tr>
      <th>511</th>
      <td>Stephen A. reacts to Kyrie Irving's Instagram Live about his stance on the vaccine | First Take</td>
      <td>[SPORTS, FIRSTTAKE, FIRSTTAKETODAY, FIRSTTAKEESPN]</td>
      <td>511</td>
    </tr>
    <tr>
      <th>512</th>
      <td>TXT (투모로우바이투게더) 'LO$ER=LO♡ER' Official MV</td>
      <td>[MUSIC, HYBE, HYBELABELS, 하이브]</td>
      <td>512</td>
    </tr>
    <tr>
      <th>513</th>
      <td>The Try Guys Make Dresses Without Instructions</td>
      <td>[COMEDY, TRYGUYS, KEITH, NED]</td>
      <td>513</td>
    </tr>
    <tr>
      <th>514</th>
      <td>Throwing My DREAM QUINCEANERA!! *emotional* | Louie's Life</td>
      <td>[ENTERTAINMENT, MYDREAM, MYDREAMQUINCE, MYDREAMQUINCEANERA]</td>
      <td>514</td>
    </tr>
    <tr>
      <th>515</th>
      <td>Toosii - head over hills (Official Music Video)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>515</td>
    </tr>
    <tr>
      <th>516</th>
      <td>Trying Clothing Hacks To See If They Work</td>
      <td>[HOWTO&amp;STYLE, SSSNIPERWOLF, LITTLELIA, LIFEHACKS]</td>
      <td>516</td>
    </tr>
    <tr>
      <th>517</th>
      <td>Updated Vote Count In Philadelphia Narrows Gap Between Biden And Trump | TODAY</td>
      <td>[NEWS&amp;POLITICS, NEWS, EDITOR'SPICKS, POLITICS]</td>
      <td>517</td>
    </tr>
    <tr>
      <th>518</th>
      <td>WE'RE BACK! | DissociaDID's OFFICIAL Return To YouTube!</td>
      <td>[EDUCATION, DID, DISSOCIATIVEIDENTITYDISORDER, MULTIPLEPERSONALITY]</td>
      <td>518</td>
    </tr>
    <tr>
      <th>519</th>
      <td>WHERE HAVE I BEEN?!</td>
      <td>[GAMING, MINECRAFT, POPULARMMOS, CHALLENGE]</td>
      <td>519</td>
    </tr>
    <tr>
      <th>520</th>
      <td>Watch Vice President Kamala Harris’ swearing in | FOX6 News Milwaukee</td>
      <td>[NEWS&amp;POLITICS, NEWS, KAMALAHARRIS, INAUGURATIONDAY]</td>
      <td>520</td>
    </tr>
    <tr>
      <th>521</th>
      <td>When A 17 Year Old Luka Doncic Had To Guard MVP Westbrook</td>
      <td>[SPORTS, LEBRONJAMES, LALAKERS, HOUSEOFHIGHLIGHTS]</td>
      <td>521</td>
    </tr>
    <tr>
      <th>522</th>
      <td>World's Smallest TV | OT 30</td>
      <td>[SPORTS, DUDEPERFECT, DUDEPERFECTSTEREOTYPES, DUDEPERFECTWATERBOTTLEFLIP]</td>
      <td>522</td>
    </tr>
    <tr>
      <th>523</th>
      <td>YBN Nahmir - Opp Stoppa (feat. 21 Savage) [Official Music Video]</td>
      <td>[MUSIC, YBN, YBNNAHMIR, YBNALMIGHTYJAY]</td>
      <td>523</td>
    </tr>
    <tr>
      <th>524</th>
      <td>[T:TIME] ‘0X1=LOVESONG (I Know I Love You) feat. Seori’ MV reaction - TXT (투모로우바이투게더)</td>
      <td>[MUSIC, 투모로우바이투게더, TOMORROWXTOGETHER, TXT]</td>
      <td>524</td>
    </tr>
    <tr>
      <th>525</th>
      <td>[최초공개] TXT (투모로우바이투게더) - No Rules (4K) | TXT COMEBACKSHOW 'FREEZE' | Mnet 210531 방송</td>
      <td>[ENTERTAINMENT, 엠넷, MNET, 엠투]</td>
      <td>525</td>
    </tr>
    <tr>
      <th>526</th>
      <td>the largest tip i've ever gotten</td>
      <td>[ENTERTAINMENT, SUBWAY, FOOD, STORY]</td>
      <td>526</td>
    </tr>
    <tr>
      <th>527</th>
      <td>#LEOMESSI: First steps and first training at the Ooredoo Center! ✔️</td>
      <td>[SPORTS, PARISSAINT-GERMAIN, PSG, PARIS]</td>
      <td>527</td>
    </tr>
    <tr>
      <th>528</th>
      <td>$45,600 Squid Game Challenge!</td>
      <td>[GAMING, None, None, None]</td>
      <td>528</td>
    </tr>
    <tr>
      <th>529</th>
      <td>10 Things NLE Choppa Can't Live Without | GQ</td>
      <td>[ENTERTAINMENT, FAVORITESTUFF, BUYERSGUIDE, WANTLIST]</td>
      <td>529</td>
    </tr>
    <tr>
      <th>530</th>
      <td>9/21/20</td>
      <td>[COMEDY, None, None, None]</td>
      <td>530</td>
    </tr>
    <tr>
      <th>531</th>
      <td>Apex Legends Fight or Fright Event 2020 Trailer</td>
      <td>[GAMING, APEXLEGENDS, APEXLEGENDSFIGHTORFRIGHT, APEXLEGENDSSHADOWROYALE]</td>
      <td>531</td>
    </tr>
    <tr>
      <th>532</th>
      <td>Aubameyang stars as Arsenal beat Liverpool on penalties to win FA Community Shield | ESPN FC</td>
      <td>[SPORTS, ARSENAL, LIVERPOOL, FACOMMUNITYSHIELD]</td>
      <td>532</td>
    </tr>
    <tr>
      <th>533</th>
      <td>BTS Butter Interview</td>
      <td>[ENTERTAINMENT, BTS, BUTTER, ZACHSANGSHOW]</td>
      <td>533</td>
    </tr>
    <tr>
      <th>534</th>
      <td>BUYING THE BEST AND WORST PRODUCTS FROM SHEIN!</td>
      <td>[COMEDY, COLLEENBALLINGER, COLLEEN, BALLINGER]</td>
      <td>534</td>
    </tr>
    <tr>
      <th>535</th>
      <td>Beating Minecraft in a 1 by 1 BARRIER</td>
      <td>[GAMING, SOCKSFOR1, BEATINGMINECRAFTINA1BY1BARRIER, MINECRAFT]</td>
      <td>535</td>
    </tr>
    <tr>
      <th>536</th>
      <td>Better | Mamba Forever | Nike</td>
      <td>[SPORTS, NIKE, NIKECOMMERCIAL, MAMBA]</td>
      <td>536</td>
    </tr>
    <tr>
      <th>537</th>
      <td>Chainsaw Carving Competition (OT 29 Outtakes)</td>
      <td>[PEOPLE&amp;BLOGS, DUDE, PERFECT, PLUS]</td>
      <td>537</td>
    </tr>
    <tr>
      <th>538</th>
      <td>Charley Pride • Country Music Hall of Fame Member • 1934-2020</td>
      <td>[MUSIC, COUNTRYMUSIC, COUNTRYMUSICHALLOFFAME, COUNTRYMUSICHALLOFFAMEANDMUSEUM]</td>
      <td>538</td>
    </tr>
    <tr>
      <th>539</th>
      <td>Davido - La La (Official Video) ft. Ckay</td>
      <td>[MUSIC, COLUMBIA, DAVIDOFEAT.CKAY, LALA]</td>
      <td>539</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Deep Rock Galactic - Season 01 - Narrated Trailer</td>
      <td>[GAMING, PC, CONSOLE, STEAM]</td>
      <td>540</td>
    </tr>
    <tr>
      <th>541</th>
      <td>Euro 2020 final: Italy and England fans react to final penalty kick</td>
      <td>[SPORTS, 2021, EURO2020, CELEBRATE]</td>
      <td>541</td>
    </tr>
    <tr>
      <th>542</th>
      <td>FRIDAY NIGHT FUNKIN - Pancake art Challenge/Agoti, Tabi, Hell Clown, Boy Friend, Ruv, Sarv, FNF, 프나펌</td>
      <td>[ENTERTAINMENT, PLAYART, PLAYARTPANCAKE, 플레이아트]</td>
      <td>542</td>
    </tr>
    <tr>
      <th>543</th>
      <td>FaZe Rug: Crimson - Official Movie Trailer</td>
      <td>[ENTERTAINMENT, FAZERUG, RUG, RUGFAZE]</td>
      <td>543</td>
    </tr>
    <tr>
      <th>544</th>
      <td>Farruko - El Incomprendido (Official Video) ft. Victor Cardenas &amp; Dj Adoni | La 167 ⛽️🏁</td>
      <td>[MUSIC, FARRUKO, FARRU, FARRUKOVICTORCARDENAS]</td>
      <td>544</td>
    </tr>
    <tr>
      <th>545</th>
      <td>Ferran OVERCOMES His BIG FEAR!! (UNEXPECTED) | The Royalty Family</td>
      <td>[PEOPLE&amp;BLOGS, THEROYALTYFAMILY, ROYALTYFAMILY, ANDREAESPADA]</td>
      <td>545</td>
    </tr>
    <tr>
      <th>546</th>
      <td>GRIZZLIES at LAKERS | FULL GAME HIGHLIGHTS | February 12, 2021</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>546</td>
    </tr>
    <tr>
      <th>547</th>
      <td>Get EA Play with Xbox Game Pass Ultimate &amp; Xbox Game Pass for PC this Holiday</td>
      <td>[GAMING, XBOX, XBOX360, XBOXONE]</td>
      <td>547</td>
    </tr>
    <tr>
      <th>548</th>
      <td>Giannis Gets Ejected After Headbutting Mo Wagner</td>
      <td>[SPORTS, BLEACHERREPORT, BR, NBA]</td>
      <td>548</td>
    </tr>
    <tr>
      <th>549</th>
      <td>Global Debut: Did Nissan Get It Right? Is The New Nissan Z a Toyota Supra Killer?</td>
      <td>[AUTOS&amp;VEHICLES, NISSANZ, NEWNISSANZCAR, NISSANZCAR]</td>
      <td>549</td>
    </tr>
    <tr>
      <th>550</th>
      <td>Haaland Scores First Two Goals in Der Klassiker - FC Bayern München vs Borussia Dortmund</td>
      <td>[SPORTS, FOOTBALL, SOCCER, BUNDESLIGA]</td>
      <td>550</td>
    </tr>
    <tr>
      <th>551</th>
      <td>How They Wrote Classic Christmas Songs</td>
      <td>[COMEDY, RYANGEORGE, CHRISTMASSONGS, WINTERWONDERLANDLYRICS]</td>
      <td>551</td>
    </tr>
    <tr>
      <th>552</th>
      <td>How To CRAFT an *INFINITY STONE SHIELD* in Minecraft (Insane Craft)</td>
      <td>[GAMING, None, None, None]</td>
      <td>552</td>
    </tr>
    <tr>
      <th>553</th>
      <td>I JOINED 1%...</td>
      <td>[GAMING, None, None, None]</td>
      <td>553</td>
    </tr>
    <tr>
      <th>554</th>
      <td>I Lost My Baby.</td>
      <td>[HOWTO&amp;STYLE, ILOSTMYBABY, PREGNANCYJOURNEY, MISCARRIAGE]</td>
      <td>554</td>
    </tr>
    <tr>
      <th>555</th>
      <td>I can't tell... are they siblings or dating..? 😳</td>
      <td>[GAMING, JANKYLANKYTV, LUDWIG, TWITCHSTREAMER]</td>
      <td>555</td>
    </tr>
    <tr>
      <th>556</th>
      <td>I watched Space Jam 2 so you don’t have to</td>
      <td>[FILM&amp;ANIMATION, ALPHARAD, SUPERSMASHBROSULTIMATE, SMASHBROS]</td>
      <td>556</td>
    </tr>
    <tr>
      <th>557</th>
      <td>If Robots Turned Evil</td>
      <td>[COMEDY, RYANGEORGE, THERYANGEORGE, SKETCH]</td>
      <td>557</td>
    </tr>
    <tr>
      <th>558</th>
      <td>Iggy Azalea - I Am The Stripclub [Official Music Video]</td>
      <td>[MUSIC, IGGYAZALEA, IAMTHESTRIPCLUB(OFFICIALVIDEO), BADDREAMSRECORDS/EMPIRE]</td>
      <td>558</td>
    </tr>
    <tr>
      <th>559</th>
      <td>KODAK IS NOT A CLONE ; The Making Of Hit Bout It | The Boat Show S2 Ep. 6</td>
      <td>[ENTERTAINMENT, KODAKBLACK, LILYACHTY, HITBOUTIT]</td>
      <td>559</td>
    </tr>
    <tr>
      <th>560</th>
      <td>Kelly Rowland - Hitman (Official Video)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>560</td>
    </tr>
    <tr>
      <th>561</th>
      <td>Kid Cudi, Skepta, Pop Smoke - Show Out (Official Visualizer)</td>
      <td>[MUSIC, KIDCUDI, CUDI, MANONTHEMOON]</td>
      <td>561</td>
    </tr>
    <tr>
      <th>562</th>
      <td>Kodak Black Ft. Rod Wave - Before I Go - [Official Music Video]</td>
      <td>[MUSIC, KODAK, BLACK, KODAKBLACK]</td>
      <td>562</td>
    </tr>
    <tr>
      <th>563</th>
      <td>LOS CAMPEONES DEL 86 DESPIDIERON A DIEGO MARADONA - TFN</td>
      <td>[NEWS&amp;POLITICS, TELEFE, NOTICIAS, TELEFENOTICIAS]</td>
      <td>563</td>
    </tr>
    <tr>
      <th>564</th>
      <td>MY LAST PREGNANCY UPDATE ON THE TWINS | OUR ULTRASOUND RESULTS</td>
      <td>[HOWTO&amp;STYLE, PREGNANCYUPDATEONTHETWINS!, UNEXPECTEDNEWS, PREGNANCYUPDATE]</td>
      <td>564</td>
    </tr>
    <tr>
      <th>565</th>
      <td>Marathon Day Is Here | How To Be Behzinga</td>
      <td>[ENTERTAINMENT, BEHZINGA, HOWTOBEBEHZINGA, ETHANPAYNE]</td>
      <td>565</td>
    </tr>
    <tr>
      <th>566</th>
      <td>Meet Ash | Apex Legends Character Trailer</td>
      <td>[GAMING, APEXLEGENDS, APEX, ASHAPEXLEGENDS]</td>
      <td>566</td>
    </tr>
    <tr>
      <th>567</th>
      <td>Messi makes his debut as a PSG player</td>
      <td>[SPORTS, FOOTBALL, FRANCEFOOTBALL, FRANCELEAGUE1]</td>
      <td>567</td>
    </tr>
    <tr>
      <th>568</th>
      <td>Minecraft Speedrunner VS 2 Assassins</td>
      <td>[GAMING, DREAMMINECRAFT, DREAMMINECRAFTYOUTUBE, MINECRAFT]</td>
      <td>568</td>
    </tr>
    <tr>
      <th>569</th>
      <td>Minecraft, But It Gets More Realistic...</td>
      <td>[GAMING, MINECRAFT, MINECRAFTBUT, MCBUT]</td>
      <td>569</td>
    </tr>
    <tr>
      <th>570</th>
      <td>Minecraft, But You Can't Touch Grass... #Shorts</td>
      <td>[FILM&amp;ANIMATION, None, None, None]</td>
      <td>570</td>
    </tr>
    <tr>
      <th>571</th>
      <td>Money or Backflip? #shorts</td>
      <td>[ENTERTAINMENT, MYSTERYBOX, IPHONEOR, IPHONEORMYSTERYBOX]</td>
      <td>571</td>
    </tr>
    <tr>
      <th>572</th>
      <td>Munich in scoring mood! Bremer SV vs. FC Bayern 0-12 | Highlights | DFB-Pokal 1. Round</td>
      <td>[SPORTS, DFB, GERMANFOOTBALL, FOOTBALL]</td>
      <td>572</td>
    </tr>
    <tr>
      <th>573</th>
      <td>Nick Jonas - Spaceman (Official Video)</td>
      <td>[MUSIC, NICK, NICKJONAS, JONASBROTHERS]</td>
      <td>573</td>
    </tr>
    <tr>
      <th>574</th>
      <td>Olivia Rodrigo - Top 18 Songs For My 18th Birthday</td>
      <td>[PEOPLE&amp;BLOGS, OLIVIARODRIGO, OLIVIA, DRIVERSLICENSE]</td>
      <td>574</td>
    </tr>
    <tr>
      <th>575</th>
      <td>Our 2 Year Anniversary Photoshoot!</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>575</td>
    </tr>
    <tr>
      <th>576</th>
      <td>Packers vs. Buccaneers Week 6 Highlights | NFL 2020</td>
      <td>[SPORTS, NFL, FOOTBALL, OFFENSE]</td>
      <td>576</td>
    </tr>
    <tr>
      <th>577</th>
      <td>Phish Dinner And A Movie Ep. 28: Hampton, VA</td>
      <td>[MUSIC, PHISH, PHISHDINNERANDAMOVIE, JAMBAND]</td>
      <td>577</td>
    </tr>
    <tr>
      <th>578</th>
      <td>Polo G - Toxic (Official Video)</td>
      <td>[MUSIC, POLOG, TOXIC, HALLOFFAME]</td>
      <td>578</td>
    </tr>
    <tr>
      <th>579</th>
      <td>Quavo Feat. Yung Miami - Strub Tha Ground (Official Video)</td>
      <td>[MUSIC, MIGOS, CITYGIRLS, QUAVO]</td>
      <td>579</td>
    </tr>
    <tr>
      <th>580</th>
      <td>Quavo Shuts Down Icebox to Shop!</td>
      <td>[ENTERTAINMENT, ICEBOX, ICEBOXOFFICIAL, ICEBOXATLANTA]</td>
      <td>580</td>
    </tr>
    <tr>
      <th>581</th>
      <td>RAPTORS at CELTICS | FULL GAME HIGHLIGHTS | September 9, 2020</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>581</td>
    </tr>
    <tr>
      <th>582</th>
      <td>Robot carves photo real pumpkins</td>
      <td>[SCIENCE&amp;TECHNOLOGY, PUMPKINCARVING, HALLOWEENPUMPKIN, JACKOLANTERN]</td>
      <td>582</td>
    </tr>
    <tr>
      <th>583</th>
      <td>Rod Wave - Street Runner (Official Video)</td>
      <td>[MUSIC, RODWAVE, HUNGERGAMES, HUNGERGAMES3]</td>
      <td>583</td>
    </tr>
    <tr>
      <th>584</th>
      <td>SOMI (전소미) - 'DUMB DUMB' M/V</td>
      <td>[MUSIC, KPOP, K-POP, SOMI]</td>
      <td>584</td>
    </tr>
    <tr>
      <th>585</th>
      <td>Secreto El Famoso Biberon - Dicen (Audio Oficial)</td>
      <td>[ENTERTAINMENT, SECRETO, EL, FAMOSO]</td>
      <td>585</td>
    </tr>
    <tr>
      <th>586</th>
      <td>Several boats sink in Central Texas Trump boat parade | KVUE</td>
      <td>[NEWS&amp;POLITICS, KVUE, AUSTIN, TRUMPBOATPARADE]</td>
      <td>586</td>
    </tr>
    <tr>
      <th>587</th>
      <td>She RUINED our FOURTH OF JULY *VERY EMOTIONAL* | The Beverly Halls</td>
      <td>[ENTERTAINMENT, BROOKEASHLEYHALL, BROOKE, BROOKEHALL]</td>
      <td>587</td>
    </tr>
    <tr>
      <th>588</th>
      <td>Skip &amp; Shannon react to Lakers dominant Game 1 win over Miami Heat in NBA Finals | NBA | UNDISPUTED</td>
      <td>[SPORTS, FOX, FOXSPORTS, FS1]</td>
      <td>588</td>
    </tr>
    <tr>
      <th>589</th>
      <td>So I made Hardcore EVEN More Difficult...</td>
      <td>[GAMING, FUNDY, FUNDYLIVE, ITSFUNDY]</td>
      <td>589</td>
    </tr>
    <tr>
      <th>590</th>
      <td>State of the Role: Revealing our Super Top Secret Project!</td>
      <td>[GAMING, CRITICALROLE, MATTMERCER, MARISHARAY]</td>
      <td>590</td>
    </tr>
    <tr>
      <th>591</th>
      <td>Stealing Ice Cream</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>591</td>
    </tr>
    <tr>
      <th>592</th>
      <td>Subscriber DESTROYS my GOLDEN PLAY BUTTON 😭 💔 ( 24 Hour Challenge)</td>
      <td>[PEOPLE&amp;BLOGS, PRETTYBOYFREDO, PRETTYBOYFREDOEXPOSED, SSH]</td>
      <td>592</td>
    </tr>
    <tr>
      <th>593</th>
      <td>Super Smash Bros. Ultimate - Mr. Sakurai Presents Sephiroth</td>
      <td>[GAMING, NINTENDO, PLAY, PLAYNINTENDO]</td>
      <td>593</td>
    </tr>
    <tr>
      <th>594</th>
      <td>THE BATMAN – Main Trailer</td>
      <td>[FILM&amp;ANIMATION, ACTION, ANDYSERKIS, BATMAN]</td>
      <td>594</td>
    </tr>
    <tr>
      <th>595</th>
      <td>The BETRAYAL that COST THEM EVERYTHING - Sea of Thieves</td>
      <td>[GAMING, SEAOFTHIEVES, WELYN, WELYNSEAOFTHIEVES]</td>
      <td>595</td>
    </tr>
    <tr>
      <th>596</th>
      <td>The End is Here...</td>
      <td>[GAMING, MARKIPLIER, UNUSANNUS, THEEND]</td>
      <td>596</td>
    </tr>
    <tr>
      <th>597</th>
      <td>The One Where I Give Birth | Labor &amp; Delivery Vlog</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>597</td>
    </tr>
    <tr>
      <th>598</th>
      <td>Transforming My Daughter Into A Cheerleader ft/ Anna McNulty</td>
      <td>[ENTERTAINMENT, JORDANMATTER, DANCEPHOTOGRAPHY, CHALLENGE]</td>
      <td>598</td>
    </tr>
    <tr>
      <th>599</th>
      <td>ULTIMATE FRENCH FRY TASTE TEST</td>
      <td>[ENTERTAINMENT, EMMACHAMBERLAIN, EMMACHAMBIE, VLOG]</td>
      <td>599</td>
    </tr>
    <tr>
      <th>600</th>
      <td>WARRIORS at MAVERICKS | FULL GAME HIGHLIGHTS | February 6, 2021</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>600</td>
    </tr>
    <tr>
      <th>601</th>
      <td>WWE 2K22 Coming March 2022!  👊💥</td>
      <td>[GAMING, WWE, 2K, TRAILER]</td>
      <td>601</td>
    </tr>
    <tr>
      <th>602</th>
      <td>We Were ALL Lied To - Daisy Ridley Confirms it</td>
      <td>[ENTERTAINMENT, STARWARSEXPALINED, STARWARSTHEORY, STARWARS]</td>
      <td>602</td>
    </tr>
    <tr>
      <th>603</th>
      <td>Weirdest Zillow Listings Part 2, Part 2 (feat. Brittany Broski) | Sarah Schauer</td>
      <td>[COMEDY, SARAHSCHAUER, SARAH, SCHAUER]</td>
      <td>603</td>
    </tr>
    <tr>
      <th>604</th>
      <td>When Your Adult Kids Live At Home</td>
      <td>[MUSIC, ADAMCALHOUN, ACAL, TAMEN]</td>
      <td>604</td>
    </tr>
    <tr>
      <th>605</th>
      <td>When the white english teacher see the n word during reading😭</td>
      <td>[COMEDY, FYP, TIKTOK’S, TIKTOKS]</td>
      <td>605</td>
    </tr>
    <tr>
      <th>606</th>
      <td>YONEX Thailand Open | Day 4: Rasmus Gemke (DEN) vs. Anthony Sinisuka Ginting (INA) [5]</td>
      <td>[SPORTS, YONEXTHAILANDOPEN, WT_2021, HSBCBWFWORLDTOURSUPER1000]</td>
      <td>606</td>
    </tr>
    <tr>
      <th>607</th>
      <td>Yungeen Ace - Giving Up (Official Music Video)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>607</td>
    </tr>
    <tr>
      <th>608</th>
      <td>i need to step away...</td>
      <td>[ENTERTAINMENT, CRANKGAMEPLAYS, CRANKGAMEPLAYS, YOUTUBEBREAK]</td>
      <td>608</td>
    </tr>
    <tr>
      <th>609</th>
      <td>iPhone 14 - EXCLUSIVE FIRST LOOK! No notch! New TITANIUM design! Don't buy iPhone 13!</td>
      <td>[SCIENCE&amp;TECHNOLOGY, APPLE, IPHONE, JONPROSSER]</td>
      <td>609</td>
    </tr>
    <tr>
      <th>610</th>
      <td>illegal minecraft rooms</td>
      <td>[GAMING, MINECRAFT, FUNNY, MINECRAFTCURSED]</td>
      <td>610</td>
    </tr>
    <tr>
      <th>611</th>
      <td>1001 WAYS TO DIE | House - Part 1</td>
      <td>[GAMING, MARKIPLIER, HOUSE, HOUSEGAME]</td>
      <td>611</td>
    </tr>
    <tr>
      <th>612</th>
      <td>18th Century Mac &amp; Cheese | Stump Sohla</td>
      <td>[ENTERTAINMENT, BINGINGWITHBABISH, BASICSWITHBABISH, COOKINGWITHBABISH]</td>
      <td>612</td>
    </tr>
    <tr>
      <th>613</th>
      <td>A Tour of Grant Imahara's Shop</td>
      <td>[SCIENCE&amp;TECHNOLOGY, TESTED, GRANTIMAHARA, GRANTIMAHARASTEAMFOUNDATION]</td>
      <td>613</td>
    </tr>
    <tr>
      <th>614</th>
      <td>AND NEWWW!!!! BIG RAMY WINS THE 2020 OLYMPIA + CHRIS BUMSTEAD REPEATS!!</td>
      <td>[SPORTS, BIGRAMY, OLYMPIAFINALS, OLYMPIARESULTS]</td>
      <td>614</td>
    </tr>
    <tr>
      <th>615</th>
      <td>ATEEZ Sing BTS, BLACKPINK, 5SOS, One Direction, And More!| 8 Bit Melody Challenge | Seventeen</td>
      <td>[HOWTO&amp;STYLE, ATEEZ, ATEEZFEVER, ATEEZMV]</td>
      <td>615</td>
    </tr>
    <tr>
      <th>616</th>
      <td>Asking Charly The Question</td>
      <td>[FILM&amp;ANIMATION, TAYLERHOLDER, TAYLORHOLDER, CHARLYJORDAN]</td>
      <td>616</td>
    </tr>
    <tr>
      <th>617</th>
      <td>Ay, DiOs Mío!</td>
      <td>[MUSIC, KAROLGSOLOTICKTOKENESPAÑOLAY, DIOSMÍO!, None]</td>
      <td>617</td>
    </tr>
    <tr>
      <th>618</th>
      <td>BUCKS at SUNS | FULL GAME 2 NBA FINALS HIGHLIGHTS | July 8, 2021</td>
      <td>[SPORTS, BASKETBALL, GLEAGUE, NBA]</td>
      <td>618</td>
    </tr>
    <tr>
      <th>619</th>
      <td>Blowing FIRE RINGS underwater</td>
      <td>[SCIENCE&amp;TECHNOLOGY, None, None, None]</td>
      <td>619</td>
    </tr>
    <tr>
      <th>620</th>
      <td>CORRUPTED (S2 P1) WHITTY ~Friday Night Funkin~ [ANIMATION]</td>
      <td>[ENTERTAINMENT, FRIDAYNIGHTFUNKINMOD, FRIDAYNIGHTFUNKIN, FNF]</td>
      <td>620</td>
    </tr>
    <tr>
      <th>621</th>
      <td>Chiefs vs. Bills Week 6 Highlights | NFL 2020</td>
      <td>[SPORTS, NFL, FOOTBALL, OFFENSE]</td>
      <td>621</td>
    </tr>
    <tr>
      <th>622</th>
      <td>DJ Khaled ft. Lil Baby &amp; Lil Durk - EVERY CHANCE I GET</td>
      <td>[MUSIC, None, None, None]</td>
      <td>622</td>
    </tr>
    <tr>
      <th>623</th>
      <td>Denis Elsa 3 p.m. update</td>
      <td>[NEWS&amp;POLITICS, ELSA, FLORIDAHURRICANE, TROPICALSTORM]</td>
      <td>623</td>
    </tr>
    <tr>
      <th>624</th>
      <td>Deploy Alien Nanites - Fortnite Week 5 Legendary Quest</td>
      <td>[GAMING, FORTNITE, ALIENNANITES, ALIENNANITESLOCATIONS]</td>
      <td>624</td>
    </tr>
    <tr>
      <th>625</th>
      <td>Destiny 2: Beyond Light - Find Truth Beyond The Tale</td>
      <td>[GAMING, DESTINY, None, None]</td>
      <td>625</td>
    </tr>
    <tr>
      <th>626</th>
      <td>Djent 2020</td>
      <td>[MUSIC, 2020, DJENT, DJENT2020]</td>
      <td>626</td>
    </tr>
    <tr>
      <th>627</th>
      <td>Driving My Wrecked Mclaren 675LT For The First Time Was A COMPLETE DISASTER</td>
      <td>[AUTOS&amp;VEHICLES, ASTONMARTIN, MERCEDES, AMG]</td>
      <td>627</td>
    </tr>
    <tr>
      <th>628</th>
      <td>EST Gee - 5500 Degrees (feat. Lil Baby, 42 Dugg, Rylo Rodriguez) [Official Audio]</td>
      <td>[MUSIC, ESTGEE, BIGGERTHANLIFEORDEATH, BTLOD]</td>
      <td>628</td>
    </tr>
    <tr>
      <th>629</th>
      <td>El Chaval De La Bachata x La Ross Maria - Estoy Perdido (Remix) Video Oficial</td>
      <td>[MUSIC, None, None, None]</td>
      <td>629</td>
    </tr>
    <tr>
      <th>630</th>
      <td>Every shoe store be like</td>
      <td>[ENTERTAINMENT, LOVELIVESERVE, LLS, LOVELIVESERVE]</td>
      <td>630</td>
    </tr>
    <tr>
      <th>631</th>
      <td>FIFA 21 | Official Career Mode Trailer</td>
      <td>[GAMING, FIFA, FIFA21, FIFA21CAREERMODE]</td>
      <td>631</td>
    </tr>
    <tr>
      <th>632</th>
      <td>Google Pixel 6 Pro vs iPhone 13 Pro CAMERA Test.</td>
      <td>[SCIENCE&amp;TECHNOLOGY, GOOGLEPIXEL, PIXEL, GOOGLE]</td>
      <td>632</td>
    </tr>
    <tr>
      <th>633</th>
      <td>Grupo Firme &amp; Lenin Ramírez - En Tu Perra Vida  - (Official Video)</td>
      <td>[MUSIC, GRUPOFIRME, REGIONALMEXICANO, None]</td>
      <td>633</td>
    </tr>
    <tr>
      <th>634</th>
      <td>HIGHLIGHTS | SPURS 2-0 ARSENAL | Son's wonder goal &amp; Kane becomes top north London derby scorer!</td>
      <td>[SPORTS, SPURS, TOTTENHAMHOTSPUR, 토트넘]</td>
      <td>634</td>
    </tr>
    <tr>
      <th>635</th>
      <td>Hatshepsut: The Forgotten Pharaoh • Puppet History</td>
      <td>[ENTERTAINMENT, PUPPETHISTORY, PUPPETHISTORYSONGS, MUSICALHISTORY]</td>
      <td>635</td>
    </tr>
    <tr>
      <th>636</th>
      <td>He did WHAT with his controller?!?!</td>
      <td>[GAMING, None, None, None]</td>
      <td>636</td>
    </tr>
    <tr>
      <th>637</th>
      <td>Historical Moment! Alyssa Wray’s Star Power Brings Lionel Richie To Tears! - American Idol 2021</td>
      <td>[ENTERTAINMENT, AMERICANIDOL, SINGINGCOMPETITION, KATYPERRY]</td>
      <td>637</td>
    </tr>
    <tr>
      <th>638</th>
      <td>Hog Hunt | Dream SMP Animation</td>
      <td>[GAMING, None, None, None]</td>
      <td>638</td>
    </tr>
    <tr>
      <th>639</th>
      <td>Honest Trailers | Mean Girls</td>
      <td>[FILM&amp;ANIMATION, SCREENJUNKIES, SCREENJUNKIES, HONESTTRAILERS]</td>
      <td>639</td>
    </tr>
    <tr>
      <th>640</th>
      <td>I Can't Believe ADELE Did This Again!</td>
      <td>[MUSIC, RICKBEATO, EVERYTHINGMUSIC, MUSIC]</td>
      <td>640</td>
    </tr>
    <tr>
      <th>641</th>
      <td>I Forced BadBoyHalo To Take a Skeppy Themed Quiz...</td>
      <td>[GAMING, MINECRAFT, MINECRAFT, MINECRAFTYOUTUBER]</td>
      <td>641</td>
    </tr>
    <tr>
      <th>642</th>
      <td>Incoming Transmission - Reality Log 474</td>
      <td>[GAMING, FORTNITE, EPICGAMES, PC]</td>
      <td>642</td>
    </tr>
    <tr>
      <th>643</th>
      <td>India take 1-0 lead after dramatic T20 opener | Dettol T20I Series 2020</td>
      <td>[SPORTS, AUSTRALIA, INDIA, FIRSTT20I]</td>
      <td>643</td>
    </tr>
    <tr>
      <th>644</th>
      <td>J. Balvin, Skrillex - In Da Getto (Official Video)</td>
      <td>[MUSIC, BALVIN, SKRILLEX, GETTO]</td>
      <td>644</td>
    </tr>
    <tr>
      <th>645</th>
      <td>JAY B - B.T.W (Feat. Jay Park) (Prod. Cha Cha Malone) (Official Video)</td>
      <td>[MUSIC, JAYB, 제이비, 박재범]</td>
      <td>645</td>
    </tr>
    <tr>
      <th>646</th>
      <td>Jack Harlow - Tyler Herro [Official Video]</td>
      <td>[MUSIC, JACKHARLOW, JACKRAPPER, HARLOWRAPPER]</td>
      <td>646</td>
    </tr>
    <tr>
      <th>647</th>
      <td>Jennifer Lopez - This Land Is Your Land &amp; America, The Beautiful - Inauguration 2021 Performance</td>
      <td>[MUSIC, JENNIFERLOPEZ, JLO, JLO]</td>
      <td>647</td>
    </tr>
    <tr>
      <th>648</th>
      <td>Julie and the Phantoms BTS | Shot Compare Edge of Great</td>
      <td>[ENTERTAINMENT, KENNYORTEGA, BOOBOOSTEWART, OWENJOYNER]</td>
      <td>648</td>
    </tr>
    <tr>
      <th>649</th>
      <td>Khabib Nurmagomedov declines Georges St-Pierre UFC fight offer, stays retired</td>
      <td>[SPORTS, MMA, MMAJUNKIE, UFC]</td>
      <td>649</td>
    </tr>
    <tr>
      <th>650</th>
      <td>Minecraft Manhunt but I can ONE HIT Hunters...</td>
      <td>[GAMING, MINECRAFT, MINECRAFT, MANHUNT]</td>
      <td>650</td>
    </tr>
    <tr>
      <th>651</th>
      <td>Minecraft, But You Can Go Inside Any Block...</td>
      <td>[GAMING, MINECRAFT, MINECRAFTBUTYOUCANGOINSIDEANYBLOCK, MINECRAFTBUT]</td>
      <td>651</td>
    </tr>
    <tr>
      <th>652</th>
      <td>Minecraft, but Eating Gives You Random Potion Effects</td>
      <td>[GAMING, MINECRAFTBUTFOODGIVESYOURANDOMPOTIONEFFECTS, MINECRAFTBUTEATINGGIVESYOUPOTIONEFFECTS, MINECRAFTBUTFOODISPOTIONS]</td>
      <td>652</td>
    </tr>
    <tr>
      <th>653</th>
      <td>Model Scout Decides Who's Most Attractive</td>
      <td>[ENTERTAINMENT, CUT, WATCHCUT, PEOPLE]</td>
      <td>653</td>
    </tr>
    <tr>
      <th>654</th>
      <td>MrBeast gave me $70,000 to do this...</td>
      <td>[ENTERTAINMENT, JACKSUCKSATLIFE, JACKSUCKSATLIFEYOUTUBE, JACKSUCKSATLIFECHANNEL]</td>
      <td>654</td>
    </tr>
    <tr>
      <th>655</th>
      <td>Napoli 2-1 Juventus | Koulibaly is the hero for the night! | Serie A 2021/22</td>
      <td>[SPORTS, RONALDO, SERIEA, DYBALA]</td>
      <td>655</td>
    </tr>
    <tr>
      <th>656</th>
      <td>Nba YoungBoy - Chopper City</td>
      <td>[MUSIC, None, None, None]</td>
      <td>656</td>
    </tr>
    <tr>
      <th>657</th>
      <td>Ohio Governor Mike DeWine - COVID-19 Statewide Address</td>
      <td>[NEWS&amp;POLITICS, None, None, None]</td>
      <td>657</td>
    </tr>
    <tr>
      <th>658</th>
      <td>Oral Roberts vs. Ohio State - First Round NCAA tournament extended highlights</td>
      <td>[SPORTS, 2021NCAAMEN'SDIVISIONIBASKETBALLTOURNAMENT(SPORTSLEAGUECHAMPIONSHIPEVENT), OHIOSTATEBUCKEYES, OHIOSTATEBUCKEYESVS.ORALROBERTSGOLDENEAGLES]</td>
      <td>658</td>
    </tr>
    <tr>
      <th>659</th>
      <td>Our Farm Got Destroyed.</td>
      <td>[PEOPLE&amp;BLOGS, FARMING, FAMILYFARM, AGRICULTURE]</td>
      <td>659</td>
    </tr>
    <tr>
      <th>660</th>
      <td>Our First Ultrasound | IT'S TWINS!</td>
      <td>[ENTERTAINMENT, OURFIRSTULTRASOUND, IT'STWINS!, FIRSTULTRASOUNDTWINS]</td>
      <td>660</td>
    </tr>
    <tr>
      <th>661</th>
      <td>Piers and Alex Clash Over Prince Harry and Meghan’s Accusations of Racism | Good Morning Britain</td>
      <td>[ENTERTAINMENT, GOODMORNINGBRITAIN, BREAKFASTSHOW, NEWS]</td>
      <td>661</td>
    </tr>
    <tr>
      <th>662</th>
      <td>PlayStation 5 Unboxing &amp; Accessories!</td>
      <td>[SCIENCE&amp;TECHNOLOGY, PS5, PLAYSTATION5, PLAYSTATION]</td>
      <td>662</td>
    </tr>
    <tr>
      <th>663</th>
      <td>Remembering Chadwick Boseman - Fat Man Beyond LIVE! 8/28/20</td>
      <td>[ENTERTAINMENT, KEVINSMITH, SMODCAST, PODCAST]</td>
      <td>663</td>
    </tr>
    <tr>
      <th>664</th>
      <td>Rest In Peace</td>
      <td>[PEOPLE&amp;BLOGS, CAMPING, CAMP, CHICKENS]</td>
      <td>664</td>
    </tr>
    <tr>
      <th>665</th>
      <td>Resumen: Ecuador 2 Bolivia 1 - Amistoso Internacional</td>
      <td>[SPORTS, ECUADOR2BOLIVIA1, RESUMEN, ELCANALDELFÚTBOL]</td>
      <td>665</td>
    </tr>
    <tr>
      <th>666</th>
      <td>Roman Reigns reveals Jimmy Uso as Daniel Bryan’s ‘replacement' | FRIDAY NIGHT SMACKDOWN</td>
      <td>[SPORTS, WWE, WWEONFOX, SMACKDOWN]</td>
      <td>666</td>
    </tr>
    <tr>
      <th>667</th>
      <td>Saints vs. Seahawks Week 7 Highlights | NFL 2021 Highlights</td>
      <td>[SPORTS, None, None, None]</td>
      <td>667</td>
    </tr>
    <tr>
      <th>668</th>
      <td>Shane's New Truck and Finished Office Reveal!</td>
      <td>[PEOPLE&amp;BLOGS, RYLANDADAMS, SHANEDAWSON, NEWCAR]</td>
      <td>668</td>
    </tr>
    <tr>
      <th>669</th>
      <td>Should Your Boyfriend Play Returnal?</td>
      <td>[GAMING, GIRLFRIENDREVIEWS, REVIEW, GAMES]</td>
      <td>669</td>
    </tr>
    <tr>
      <th>670</th>
      <td>So He CHEATED And BURNED My Minecraft House... (troll)</td>
      <td>[GAMING, SLOGOMAN, GAMING, MINECRAFT]</td>
      <td>670</td>
    </tr>
    <tr>
      <th>671</th>
      <td>Stop using me for click bait! 🙄</td>
      <td>[ENTERTAINMENT, THEREALHOUSEWIVESOFATLANTA, NENELEAKES, CYNTHIABAILEY]</td>
      <td>671</td>
    </tr>
    <tr>
      <th>672</th>
      <td>Swapping Outfits With My Best Friend!</td>
      <td>[COMEDY, SWAPPINGOUTFITSWITHMYBESTFRIEND, BESTFRIEND, SWAPPINGOUTFITSWITHJAMESCHARLES]</td>
      <td>672</td>
    </tr>
    <tr>
      <th>673</th>
      <td>TEACHING BRETMAN ROCK HOW TO PLAY ROBLOX</td>
      <td>[PEOPLE&amp;BLOGS, LARRAY, BRETMAN, ROBLOX]</td>
      <td>673</td>
    </tr>
    <tr>
      <th>674</th>
      <td>The Fortnite Neymar Jr Outfit Cinematic Reveal Trailer</td>
      <td>[GAMING, YT:CC=ON, None, None]</td>
      <td>674</td>
    </tr>
    <tr>
      <th>675</th>
      <td>This MUST Be Fake</td>
      <td>[SCIENCE&amp;TECHNOLOGY, INTEL, GAMING, CPU]</td>
      <td>675</td>
    </tr>
    <tr>
      <th>676</th>
      <td>Toosii - shop (Official Video) ft. DaBaby</td>
      <td>[MUSIC, TOOSII, TOOSII2X, TOOSIILOVECYCLE]</td>
      <td>676</td>
    </tr>
    <tr>
      <th>677</th>
      <td>Trying The Squid Game Honeycomb Candy Challenge</td>
      <td>[ENTERTAINMENT, SSSNIPERWOLF, SNIPERWOLF, REACTING]</td>
      <td>677</td>
    </tr>
    <tr>
      <th>678</th>
      <td>UFC 253 Embedded: Vlog Series - Episode 3</td>
      <td>[SPORTS, UFC, ULTIMATE, FIGHTING]</td>
      <td>678</td>
    </tr>
    <tr>
      <th>679</th>
      <td>UFC 266 Embedded: Vlog Series - Episode 3</td>
      <td>[SPORTS, UFC, 266, UFC266]</td>
      <td>679</td>
    </tr>
    <tr>
      <th>680</th>
      <td>Venom – The Birth of Carnage, A PlayStation Exclusive Extended Sneak Peek</td>
      <td>[FILM&amp;ANIMATION, VENOM:LETTHEREBECARNAGE, VENOM, LETTHEREBECARNAGE]</td>
      <td>680</td>
    </tr>
    <tr>
      <th>681</th>
      <td>WARRIORS at NETS | FULL GAME HIGHLIGHTS | December 22, 2020</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>681</td>
    </tr>
    <tr>
      <th>682</th>
      <td>What Really Happened at West Coast Customs | Up to Speed</td>
      <td>[AUTOS&amp;VEHICLES, None, None, None]</td>
      <td>682</td>
    </tr>
    <tr>
      <th>683</th>
      <td>When you can't tell if you're in the Friendzone...</td>
      <td>[ENTERTAINMENT, FRIENDZONE, WHENYOUDON'TKNOWIFYOU'REINTHEFRIENDZONE, INTHEFRIENDZONE]</td>
      <td>683</td>
    </tr>
    <tr>
      <th>684</th>
      <td>Yoga For A Fresh Start  |  Yoga With Adriene</td>
      <td>[HOWTO&amp;STYLE, HOMEYOGA, HOMEYOGAPRACTICE, YOGAATHOME]</td>
      <td>684</td>
    </tr>
    <tr>
      <th>685</th>
      <td>iOS 14 - 17 Settings You NEED to Change Immediately!</td>
      <td>[SCIENCE&amp;TECHNOLOGY, IOS14, IOS14SETTINGS, IOS14TRICKS]</td>
      <td>685</td>
    </tr>
    <tr>
      <th>686</th>
      <td>iOS 14 iPhone home screen customization + widgets/app icons! ✨*EASY HOW-TO*</td>
      <td>[EDUCATION, IOS14, WIDGETSMITH, IPHONECUSTOMIZATION]</td>
      <td>686</td>
    </tr>
    <tr>
      <th>687</th>
      <td>valentines chocolate</td>
      <td>[GAMING, ZELDA, LINK, BOTW]</td>
      <td>687</td>
    </tr>
    <tr>
      <th>688</th>
      <td>*NEW* CRYPTO Mod in Among Us</td>
      <td>[GAMING, None, None, None]</td>
      <td>688</td>
    </tr>
    <tr>
      <th>689</th>
      <td>100 Things You Should NEVER Do In Minecraft</td>
      <td>[GAMING, LOVERFELLA, LOVERFELLASERVER, None]</td>
      <td>689</td>
    </tr>
    <tr>
      <th>690</th>
      <td>42 Dugg - FREE RIC (feat. Lil Durk) [Official Music Video]</td>
      <td>[MUSIC, DUGG, FREE, RIC]</td>
      <td>690</td>
    </tr>
    <tr>
      <th>691</th>
      <td>50 Hours Inside the Most Radioactive Place On Earth (Exclusion Zone of Chernobyl)</td>
      <td>[ENTERTAINMENT, YESTHEORY, SEEKDISCOMFORT, YESTHEORYSTRANGERS]</td>
      <td>691</td>
    </tr>
    <tr>
      <th>692</th>
      <td>6.0 Sierra Nevada Earthquake Felt Across Bay Area</td>
      <td>[NEWS&amp;POLITICS, EARTHQUAKE, SIERRANEVADA, CALIFORNIA]</td>
      <td>692</td>
    </tr>
    <tr>
      <th>693</th>
      <td>A Crap Guide</td>
      <td>[ENTERTAINMENT, JOCAT, HIJEK, CRAPGUIDE]</td>
      <td>693</td>
    </tr>
    <tr>
      <th>694</th>
      <td>A Plane Without Wings: The Story of The C.450 Coléoptère</td>
      <td>[EDUCATION, SNECMACOLÉOPTÈRE, C.450COLEOPTERE, VERTICALTAKE-OFFANDLANDING(VTOL)AIRCRAFT]</td>
      <td>694</td>
    </tr>
    <tr>
      <th>695</th>
      <td>Among Us - BOSS ALIEN impostor</td>
      <td>[GAMING, ALMAZAMONGUS, ALMAZANIMATION, AMONGUS]</td>
      <td>695</td>
    </tr>
    <tr>
      <th>696</th>
      <td>Banned From Garry's Mod UK</td>
      <td>[GAMING, FUNNY, TWITCH, HIGHLIGHTS]</td>
      <td>696</td>
    </tr>
    <tr>
      <th>697</th>
      <td>Best Supporting Actors</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>697</td>
    </tr>
    <tr>
      <th>698</th>
      <td>Big Sean - Deep Reverence (Audio) ft. Nipsey Hussle</td>
      <td>[MUSIC, BIG, SEAN, DEEP]</td>
      <td>698</td>
    </tr>
    <tr>
      <th>699</th>
      <td>Billie Eilish - Oxytocin (From Disney’s Happier Than Ever: A Love Letter To Los Angeles)</td>
      <td>[MUSIC, BILLIE, EILISH, OXYTOCIN]</td>
      <td>699</td>
    </tr>
    <tr>
      <th>700</th>
      <td>Bruno Mars, Anderson .Paak, Silk Sonic - Skate [Official Music Video]</td>
      <td>[MUSIC, BRUNOMARS, BRUNO, SKATE]</td>
      <td>700</td>
    </tr>
    <tr>
      <th>701</th>
      <td>Chicago rapper 'KTS Dre' shot more than 60 times outside Cook County Jail</td>
      <td>[NEWS&amp;POLITICS, VIDEO, NEWS, WGNTVNEWS]</td>
      <td>701</td>
    </tr>
    <tr>
      <th>702</th>
      <td>Chicago rapper Lil Reese among 3 shot in parking garage</td>
      <td>[NEWS&amp;POLITICS, VIDEO, NEWS, WGNTVNEWS]</td>
      <td>702</td>
    </tr>
    <tr>
      <th>703</th>
      <td>Clash Royale: The Summer Update Is Arriving! 🏆☀️ (TV Royale)</td>
      <td>[GAMING, CLASHROYALE, CLASHROYALEGAME, SUPERCELL]</td>
      <td>703</td>
    </tr>
    <tr>
      <th>704</th>
      <td>Colbert Nyambayar HIGHLIGHTS: July 3, 2021 | PBC on SHOWTIME</td>
      <td>[SPORTS, BOXING, SPORTS, TV]</td>
      <td>704</td>
    </tr>
    <tr>
      <th>705</th>
      <td>Courtney's Boyfriend Reveal!</td>
      <td>[COMEDY, SMOSH, SMOSHPIT, SMOSHGAMES]</td>
      <td>705</td>
    </tr>
    <tr>
      <th>706</th>
      <td>Crew-1 Mission | Return</td>
      <td>[SCIENCE&amp;TECHNOLOGY, SPACEX, SPACE, MUSK]</td>
      <td>706</td>
    </tr>
    <tr>
      <th>707</th>
      <td>DEJI VS VINNIE HACKER – SOCIAL GLOVES WEIGH-IN</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>707</td>
    </tr>
    <tr>
      <th>708</th>
      <td>DJ Snake, Ozuna, Megan Thee Stallion, LISA of BLACKPINK - SG (Official Music Video)</td>
      <td>[MUSIC, SNAKE, OZUNA, MEGAN]</td>
      <td>708</td>
    </tr>
    <tr>
      <th>709</th>
      <td>Dear Baby... | Our Rainbow Pregnancy After 6 Years of Infertility</td>
      <td>[PEOPLE&amp;BLOGS, PREGNANCYANNOUNCEMENT, BABYANNOUNCEMENT, TTCJOURNEYBABY#1]</td>
      <td>709</td>
    </tr>
    <tr>
      <th>710</th>
      <td>Dolphins vs. Raiders Week 16 Highlights | NFL 2020</td>
      <td>[SPORTS, NFL, FOOTBALL, OFFENSE]</td>
      <td>710</td>
    </tr>
    <tr>
      <th>711</th>
      <td>Drag Queens Official Video HaZel aka Monique Samuels</td>
      <td>[PEOPLE&amp;BLOGS, DRAGQUEENS, DRAG, RUPAUL'SDRAGRACE]</td>
      <td>711</td>
    </tr>
    <tr>
      <th>712</th>
      <td>ENHYPEN (엔하이픈) 'Drunk-Dazed' Official MV</td>
      <td>[MUSIC, BIGHIT, 빅히트, 방탄소년단]</td>
      <td>712</td>
    </tr>
    <tr>
      <th>713</th>
      <td>Eladio Carrión - Cuarentena (Video Oficial)</td>
      <td>[MUSIC, ELADIO, CARRION, SEN2KBRN]</td>
      <td>713</td>
    </tr>
    <tr>
      <th>714</th>
      <td>Enter the World of Tzeentch | Total War: WARHAMMER III</td>
      <td>[GAMING, TOTALWAR, WARHAMMER, TOTALWARWARHAMMER]</td>
      <td>714</td>
    </tr>
    <tr>
      <th>715</th>
      <td>Every Buffalo Wild Wings Ever</td>
      <td>[COMEDY, SMOSH, SMOSHPIT, SMOSHGAMES]</td>
      <td>715</td>
    </tr>
    <tr>
      <th>716</th>
      <td>Food Theory: Yes, The Knock Off Is BETTER!</td>
      <td>[HOWTO&amp;STYLE, KNOCKOFF, KNOCKOFF, KNOCKOFFFOODBRANDS]</td>
      <td>716</td>
    </tr>
    <tr>
      <th>717</th>
      <td>GOLD iPhone 12 Pro Unboxing! Unlike the Others?</td>
      <td>[SCIENCE&amp;TECHNOLOGY, GADGETMATCH, GADGETMATCH, MICHAELJOSH]</td>
      <td>717</td>
    </tr>
    <tr>
      <th>718</th>
      <td>GTA 5 Just For Fun #15 #shorts #GTA5</td>
      <td>[GAMING, None, None, None]</td>
      <td>718</td>
    </tr>
    <tr>
      <th>719</th>
      <td>GTA 5 Online - NEW LOS SANTOS TUNERS Update - New Cars, Garage, Meet, Missions &amp; MORE!</td>
      <td>[GAMING, GTA5, GTAONLINE, GTA]</td>
      <td>719</td>
    </tr>
    <tr>
      <th>720</th>
      <td>HIGHLIGHTS | Leicester City vs Arsenal (0-2) | Carabao Cup</td>
      <td>[SPORTS, ARSENAL, ARSENALFC, ARSENALFOOTBALLCLUB]</td>
      <td>720</td>
    </tr>
    <tr>
      <th>721</th>
      <td>HSBC BWF World Tour Finals | Day 5: Lee/Wang (TPE) vs. Ahsan/Setiawan (INA)</td>
      <td>[SPORTS, #HSBCBADMINTON, #BWF, #BADMINTON]</td>
      <td>721</td>
    </tr>
    <tr>
      <th>722</th>
      <td>High School Theater</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>722</td>
    </tr>
    <tr>
      <th>723</th>
      <td>Honeykomb Brazy - Brazy Sh#t ( Mirror Flow full)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>723</td>
    </tr>
    <tr>
      <th>724</th>
      <td>How Everybody Calling Kanye after his Album Dropped</td>
      <td>[COMEDY, None, None, None]</td>
      <td>724</td>
    </tr>
    <tr>
      <th>725</th>
      <td>How a Chinese Barbecue Master Has Been Roasting Whole Pigs for 30 Years — Smoke Point</td>
      <td>[HOWTO&amp;STYLE, CHINESEBARBECUEPORK, CHINESEBARBECUE, CHINESEBBQ]</td>
      <td>725</td>
    </tr>
    <tr>
      <th>726</th>
      <td>Hurricane Laura Makes Landfall | LIVE Coverage on The Weather Channel</td>
      <td>[NEWS&amp;POLITICS, None, None, None]</td>
      <td>726</td>
    </tr>
    <tr>
      <th>727</th>
      <td>I Made A Giant 100-Pound Boba Milk Tea • Tasty</td>
      <td>[HOWTO&amp;STYLE, TASTY, TASTYRECIPES, BUZZFEED]</td>
      <td>727</td>
    </tr>
    <tr>
      <th>728</th>
      <td>I Was In The Funniest Minecraft Competition With Dream</td>
      <td>[ENTERTAINMENT, QUACKITY, QUACKITWO, QUACKITYSECONDCHANNEL]</td>
      <td>728</td>
    </tr>
    <tr>
      <th>729</th>
      <td>I got the LONGEST Username Possible on a REAL Youtube Silver Play Button</td>
      <td>[ENTERTAINMENT, JACKSUCKSATLIFE, JACKSUCKSATLIFEYOUTUBE, JACKSUCKSATLIFEYOUTUBECHANNEL]</td>
      <td>729</td>
    </tr>
    <tr>
      <th>730</th>
      <td>Inside Mayweather vs. Paul | Full Episode (TV14) | SHOWTIME PPV</td>
      <td>[SPORTS, SHOWTIME, SHOSPORTS, SPORTS]</td>
      <td>730</td>
    </tr>
    <tr>
      <th>731</th>
      <td>Justin Gaethje discusses loss to Khabib Nurmagomedov | UFC 254 Post Show | ESPN MMA</td>
      <td>[SPORTS, JUSTINGAETHJE, JUSTINGAETHJEUFC254, KHABIBNURMAGOMEDOV]</td>
      <td>731</td>
    </tr>
    <tr>
      <th>732</th>
      <td>KSI – No Time (feat. Lil Durk) [Official Video]</td>
      <td>[MUSIC, KSI, LILDURK, LILDURKKSI]</td>
      <td>732</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Khloé Kardashian's Heartbreaking COVID Quarantine Without Daughter True</td>
      <td>[ENTERTAINMENT, ELLEN, THEELLENSHOW, ELLENDEGENERES]</td>
      <td>733</td>
    </tr>
    <tr>
      <th>734</th>
      <td>King Von - Demon (Official Video)</td>
      <td>[MUSIC, KINGVON, DEMON(OFFICIALVIDEO), ONLYTHEFAMILYENTERTAINMENT/EMPIRE]</td>
      <td>734</td>
    </tr>
    <tr>
      <th>735</th>
      <td>Lil Durk - Finesse Out The Gang Way feat. Lil Baby (Official Lyric Video)</td>
      <td>[MUSIC, LILDURK, LILDURKMUSIC, LILDURKMUSICVIDEO]</td>
      <td>735</td>
    </tr>
    <tr>
      <th>736</th>
      <td>Local fallout continues following DC Capitol riots</td>
      <td>[NEWS&amp;POLITICS, VIDEO, NEWS, WGNTVNEWS]</td>
      <td>736</td>
    </tr>
    <tr>
      <th>737</th>
      <td>MY MIU MIU MAKEOVER FOR FASHION WEEK!!!</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>737</td>
    </tr>
    <tr>
      <th>738</th>
      <td>Making A Valentine's  Day Dress!</td>
      <td>[ENTERTAINMENT, TWEEZING, NICE, UM]</td>
      <td>738</td>
    </tr>
    <tr>
      <th>739</th>
      <td>Megan Fox Tries To Stop Machine Gun Kelly &amp; Conor McGregor’s Fight At MTV VMAs</td>
      <td>[ENTERTAINMENT, MEGANFOX, MACHINEGUNKELLY, CONORMCGREGOR]</td>
      <td>739</td>
    </tr>
    <tr>
      <th>740</th>
      <td>Minecraft, But Enchants Are OP...</td>
      <td>[GAMING, MINECRAFT, COMPETITIVEMINECRAFT, MINECRAFTPVP]</td>
      <td>740</td>
    </tr>
    <tr>
      <th>741</th>
      <td>Moneybagg Yo - If Pain Was A Person (Official Audio)</td>
      <td>[MUSIC, MONEYBAGGYO, MONEYBAGYO, IFPAINWASAPERSON]</td>
      <td>741</td>
    </tr>
    <tr>
      <th>742</th>
      <td>Naruto Stops Kawaki! | Boruto: Naruto Next Generations</td>
      <td>[FILM&amp;ANIMATION, CRUNCHYROLL, ANIME, ANIMETRAILER]</td>
      <td>742</td>
    </tr>
    <tr>
      <th>743</th>
      <td>Northern California Storm Updates | Monday Morning</td>
      <td>[NEWS&amp;POLITICS, SACRAMENTO, STORMWATCH, CREEK]</td>
      <td>743</td>
    </tr>
    <tr>
      <th>744</th>
      <td>Pandemic Game Night - SNL</td>
      <td>[ENTERTAINMENT, SNL, SATURDAYNIGHTLIVE, SEASON46]</td>
      <td>744</td>
    </tr>
    <tr>
      <th>745</th>
      <td>Pikete - Nicky Jam x El Alfa | Video Oficial</td>
      <td>[MUSIC, NICKYJAM, NICKYJAMPR, PIENSASENMI]</td>
      <td>745</td>
    </tr>
    <tr>
      <th>746</th>
      <td>Playing Among Us In Real Life 2!</td>
      <td>[ENTERTAINMENT, JAMES, JAMESCHARLES, MAKEUPARTIST]</td>
      <td>746</td>
    </tr>
    <tr>
      <th>747</th>
      <td>Prince Harry Sent 'Deeply Personal' Letter to Prince Charles | Lorraine</td>
      <td>[ENTERTAINMENT, LORRAINE, LORRAINEKELLY, LORRAINEITV]</td>
      <td>747</td>
    </tr>
    <tr>
      <th>748</th>
      <td>Rating Celebrities I've met at Parties/Events</td>
      <td>[FILM&amp;ANIMATION, SWOOZIE, ADANDE, ANIMATION]</td>
      <td>748</td>
    </tr>
    <tr>
      <th>749</th>
      <td>Reaction after Cristiano Ronaldo scores twice on his Manchester United return</td>
      <td>[SPORTS, SKYSPORTSNEWS, SKYSPORTS, SKYSPORTSFOOTBALL]</td>
      <td>749</td>
    </tr>
    <tr>
      <th>750</th>
      <td>Surprising My Friends with a BILLBOARD + $25,000 GIVEAWAY!!</td>
      <td>[ENTERTAINMENT, ALISHAMARIE, ALISHA, ALISHAMARIE]</td>
      <td>750</td>
    </tr>
    <tr>
      <th>751</th>
      <td>TIKTOK in a Mexican Household [Part 2]</td>
      <td>[COMEDY, TIKTOKINAMEXICANHOUSEHOLD, TIKTOK, MEXICANHOUSEHOLD]</td>
      <td>751</td>
    </tr>
    <tr>
      <th>752</th>
      <td>Tainy, Bad Bunny, Julieta Venegas - Lo Siento BB:/ (Official Video)</td>
      <td>[MUSIC, TAINY, BAD, BUNNY]</td>
      <td>752</td>
    </tr>
    <tr>
      <th>753</th>
      <td>The Fortnite UFO Experience</td>
      <td>[GAMING, FORTNITE, DANTDM, DANTDMLIVE]</td>
      <td>753</td>
    </tr>
    <tr>
      <th>754</th>
      <td>The Funniest Bloopers From FEAR STREET | Netflix</td>
      <td>[ENTERTAINMENT, CHILLING, CURSES, HIGHSCHOOL]</td>
      <td>754</td>
    </tr>
    <tr>
      <th>755</th>
      <td>This is the end…</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>755</td>
    </tr>
    <tr>
      <th>756</th>
      <td>Try Not To Laugh Challenge #65 - Everything is Connected!</td>
      <td>[PEOPLE&amp;BLOGS, SMOSH, SMOSHPIT, SMOSHGAMES]</td>
      <td>756</td>
    </tr>
    <tr>
      <th>757</th>
      <td>Vent | Among Us Animation</td>
      <td>[FILM&amp;ANIMATION, 세치혀, 끼룩, 애니메이션]</td>
      <td>757</td>
    </tr>
    <tr>
      <th>758</th>
      <td>WANDAVISION Episode 7 REACTION! Ending Explained &amp; Agnes' Plan! | Inside Marvel</td>
      <td>[ENTERTAINMENT, NEWROCKSTARS, WANDAVISION, WANDAVISIONEPISODE]</td>
      <td>758</td>
    </tr>
    <tr>
      <th>759</th>
      <td>We Bought 3 Cheap Ferraris For The Price Of A Toyota Camry - Car Trek S4E1</td>
      <td>[AUTOS&amp;VEHICLES, ASTONMARTIN, MERCEDES, AMG]</td>
      <td>759</td>
    </tr>
    <tr>
      <th>760</th>
      <td>What If You Were Stranded In the Sahara Alone?</td>
      <td>[EDUCATION, REALLIFELORE, REALLIFELOREMAPS, REALLIFELOREGEOGRAPHY]</td>
      <td>760</td>
    </tr>
    <tr>
      <th>761</th>
      <td>What if your Car Falls Off a Trailer on The Highway?</td>
      <td>[AUTOS&amp;VEHICLES, CAMARO, CORVETTE, LS2]</td>
      <td>761</td>
    </tr>
    <tr>
      <th>762</th>
      <td>Yung Bleu - Way More Close (Stuck In A Box) [Official Video) ft. Big Sean</td>
      <td>[MUSIC, YUNG, BLEU, YUNGBLEUMUSIC]</td>
      <td>762</td>
    </tr>
    <tr>
      <th>763</th>
      <td>[2021 FESTA] BTS (방탄소년단) BTS ROOM LIVE #2021BTSFESTA</td>
      <td>[MUSIC, 방탄소년단, BTS, BANGTAN]</td>
      <td>763</td>
    </tr>
    <tr>
      <th>764</th>
      <td>girl in red - rue (official video)</td>
      <td>[ENTERTAINMENT, GIRLINREDLYRICS, GIRLINREDDEADGIRLINTHEPOOL, GIRLINREDGIRLS]</td>
      <td>764</td>
    </tr>
    <tr>
      <th>765</th>
      <td>iOS 14 is Out! - What's New?</td>
      <td>[SCIENCE&amp;TECHNOLOGY, IOS14, IOS14ISOUT, IOS14RELEASED]</td>
      <td>765</td>
    </tr>
    <tr>
      <th>766</th>
      <td>iPad Mini 2021 Review: Pocketable Power!</td>
      <td>[SCIENCE&amp;TECHNOLOGY, IPADMINI, IPADMINIREVIEW, MKBHD]</td>
      <td>766</td>
    </tr>
    <tr>
      <th>767</th>
      <td>iPhone 12 Pro review: Pros and cons</td>
      <td>[SCIENCE&amp;TECHNOLOGY, TOMSGUIDE, TECH, NEWS]</td>
      <td>767</td>
    </tr>
    <tr>
      <th>768</th>
      <td>iPhone 13 Pro Max Drop Test! Heaviest iPhone Ever</td>
      <td>[SCIENCE&amp;TECHNOLOGY, IPHONE13, IPHONE13PROMAX, IPHONE13PROMAXDROPTEST]</td>
      <td>768</td>
    </tr>
    <tr>
      <th>769</th>
      <td>telling our friends &amp; family we are having baby #2!! pregnancy announcement!</td>
      <td>[PEOPLE&amp;BLOGS, ASPYNVLOGPLAYLIST, ASPYNOVARD, ASPYNANDPARKER]</td>
      <td>769</td>
    </tr>
    <tr>
      <th>770</th>
      <td>#3 BUCKS at #2 NETS | FULL GAME HIGHLIGHTS | June 5, 2021</td>
      <td>[SPORTS, BASKETBALL, GLEAGUE, NBA]</td>
      <td>770</td>
    </tr>
    <tr>
      <th>771</th>
      <td>73 Questions With Cole Sprouse | Vogue</td>
      <td>[ENTERTAINMENT, 73QSWITHCOLESPROUSE, 73QUESTIONS, 73QUESTIONSWITHCOLESPROUSE]</td>
      <td>771</td>
    </tr>
    <tr>
      <th>772</th>
      <td>All *NEW* Leaked Skins &amp; Emotes! *SUICIDE SQUAD* (Human Bill, Bloodsport, Ariana Grande)</td>
      <td>[GAMING, FORTNITE, FORTNITELEAKS, FORTNITEHYPEX]</td>
      <td>772</td>
    </tr>
    <tr>
      <th>773</th>
      <td>Among Us But Its A Reality Show 3</td>
      <td>[COMEDY, LAUGHOVERLIFE, AMONGUSBUTITSAREALITYSHOW, AMONGUS]</td>
      <td>773</td>
    </tr>
    <tr>
      <th>774</th>
      <td>Apple's entire iOS 15 event in 11 minutes (WWDC21 supercut)</td>
      <td>[SCIENCE&amp;TECHNOLOGY, CNET, TECHNOLOGY, NEWS]</td>
      <td>774</td>
    </tr>
    <tr>
      <th>775</th>
      <td>Basketball Interviews Gone Wrong - Key &amp; Peele</td>
      <td>[COMEDY, KEY&amp;PEELE, KEYANDPEELE, JORDANPEELE]</td>
      <td>775</td>
    </tr>
    <tr>
      <th>776</th>
      <td>Bryant Myers - Se Fuerte</td>
      <td>[MUSIC, BRYANT, MYERS, FUERTE]</td>
      <td>776</td>
    </tr>
    <tr>
      <th>777</th>
      <td>Burna Boy - Monsters You Made [Official Music Video]</td>
      <td>[MUSIC, BURNABOY, BURNERBOY, BURNABOY]</td>
      <td>777</td>
    </tr>
    <tr>
      <th>778</th>
      <td>Countdown Vampires (PS1) - Angry Video Game Nerd (AVGN)</td>
      <td>[GAMING, ANGRYVIDEOGAMENERD, AVGN, AVGNPS1]</td>
      <td>778</td>
    </tr>
    <tr>
      <th>779</th>
      <td>DaBaby - Blind feat Young Thug (Official Audio)</td>
      <td>[MUSIC, DABABY, BABYJESUS, CHARLOTTE]</td>
      <td>779</td>
    </tr>
    <tr>
      <th>780</th>
      <td>Deck the Halls and Auld Lang Syne Medley (feat., Ni/Co) l The Great Gift Exchange!</td>
      <td>[ENTERTAINMENT, GIFTWRAP, HOLIDAYS, ALTONDULANEY]</td>
      <td>780</td>
    </tr>
    <tr>
      <th>781</th>
      <td>Disney’s Cruella | Sneak Peek</td>
      <td>[FILM&amp;ANIMATION, None, None, None]</td>
      <td>781</td>
    </tr>
    <tr>
      <th>782</th>
      <td>Dream And Sapnap Have A Western Showdown...</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>782</td>
    </tr>
    <tr>
      <th>783</th>
      <td>Dream, I Am Your Father...</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>783</td>
    </tr>
    <tr>
      <th>784</th>
      <td>ENEMIGOS OCULTOS REMIX - OZUNA X T.Y.S X ROCHY X WILMER ROBERTS X SHELOW SHAQ X MUSICOLOGO X OMEGA</td>
      <td>[MUSIC, None, None, None]</td>
      <td>784</td>
    </tr>
    <tr>
      <th>785</th>
      <td>EPIC HOMEMADE James Bond Car</td>
      <td>[ENTERTAINMENT, COLIN, FURZE, JAMESBOND]</td>
      <td>785</td>
    </tr>
    <tr>
      <th>786</th>
      <td>F-150 Lightning Live Reveal | F-150 | Ford</td>
      <td>[AUTOS&amp;VEHICLES, CAR, AUTOMOBILE, None]</td>
      <td>786</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Foolio - JWET (Official Music Video)</td>
      <td>[MUSIC, FOOLIO, JULIOFOOLIO, FOOLIOFEELIT]</td>
      <td>787</td>
    </tr>
    <tr>
      <th>788</th>
      <td>French Montana - Hot Boy Bling ft. Jack Harlow &amp; Lil Durk [Official Video]</td>
      <td>[MUSIC, LATIN, LATINO, LATINA]</td>
      <td>788</td>
    </tr>
    <tr>
      <th>789</th>
      <td>Friday Night Funkin' Logic, But ANIME | Cartoon Animation</td>
      <td>[FILM&amp;ANIMATION, FRIDAYNIGHTFUNKIN, FNF, FNFLOGIC]</td>
      <td>789</td>
    </tr>
    <tr>
      <th>790</th>
      <td>GTA 5 Roleplay - BATMAN JET HITMAN JOBS | RedlineRP</td>
      <td>[GAMING, GTA5, GTA5ROLEPLAY, HITMAN]</td>
      <td>790</td>
    </tr>
    <tr>
      <th>791</th>
      <td>Game Theory: FNAF, Return To The Pit (3 New FNAF Theories)</td>
      <td>[GAMING, FNAF, FIVENIGHTSATFREDDY'S, FAZBEARFRIGHTS]</td>
      <td>791</td>
    </tr>
    <tr>
      <th>792</th>
      <td>Guessing YouTubers Using Only Their Oldest Video!</td>
      <td>[ENTERTAINMENT, FUNNY, CHALLENGE, FAMILY]</td>
      <td>792</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Half of her hair had to GO!</td>
      <td>[FILM&amp;ANIMATION, HAIR, HAIRCARE, BLACKHAIR]</td>
      <td>793</td>
    </tr>
    <tr>
      <th>794</th>
      <td>Hermitcraft 8 | Ep 03: SUPER SIMPLE MOB FARM!</td>
      <td>[GAMING, MINECRAFTHERMITCRAFT, HERMITCRAFT, LETSPLAYMINECRAFT]</td>
      <td>794</td>
    </tr>
    <tr>
      <th>795</th>
      <td>Hulk VS Broly (Marvel VS Dragon Ball) | DEATH BATTLE!</td>
      <td>[ENTERTAINMENT, DEATHBATTLE, ROOSTERTEETH, RT]</td>
      <td>795</td>
    </tr>
    <tr>
      <th>796</th>
      <td>Hurricane Ida vs. Hurricane Katrina: Comparing the two storms</td>
      <td>[NEWS&amp;POLITICS, HURRICANEIDA, HURRICANEIDATRACK, IDAPATH]</td>
      <td>796</td>
    </tr>
    <tr>
      <th>797</th>
      <td>I Bought A Running BMW For $600 (Because It's FULL OF ROACHES)</td>
      <td>[AUTOS&amp;VEHICLES, ASTONMARTIN, MERCEDES, AMG]</td>
      <td>797</td>
    </tr>
    <tr>
      <th>798</th>
      <td>I Gave People $1,000,000 But ONLY 1 Minute To Spend It!</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>798</td>
    </tr>
    <tr>
      <th>799</th>
      <td>I Got Hunted By A Bounty Hunter</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>799</td>
    </tr>
    <tr>
      <th>800</th>
      <td>I Paid A Stranger On The Internet To Be My Friend For 24hrs</td>
      <td>[ENTERTAINMENT, YESTHEORY, SEEKDISCOMFORT, YESTHEORYSTRANGER]</td>
      <td>800</td>
    </tr>
    <tr>
      <th>801</th>
      <td>I have 3 names</td>
      <td>[HOWTO&amp;STYLE, HAYASHIRICE, DEMIGLACESAUCE, OMURICE]</td>
      <td>801</td>
    </tr>
    <tr>
      <th>802</th>
      <td>I'm Moving Away... (not a prank)</td>
      <td>[ENTERTAINMENT, BENAZELART, BRENTRIVERA, LEXIRIVERA]</td>
      <td>802</td>
    </tr>
    <tr>
      <th>803</th>
      <td>I'm leaving Funhaus... Sort of.</td>
      <td>[PEOPLE&amp;BLOGS, FUNHAUS, ROOSTERTEETH, MOVINGON]</td>
      <td>803</td>
    </tr>
    <tr>
      <th>804</th>
      <td>I've never held so much graphics POWAHHH</td>
      <td>[SCIENCE&amp;TECHNOLOGY, NVIDIA, RTX, 3000]</td>
      <td>804</td>
    </tr>
    <tr>
      <th>805</th>
      <td>Insane Balloon Challenge!</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>805</td>
    </tr>
    <tr>
      <th>806</th>
      <td>Introducing The World's First STREET LEGAL Lamborghini Super Trofeo Evo - Unicorn V4 [4K UHD]</td>
      <td>[AUTOS&amp;VEHICLES, None, None, None]</td>
      <td>806</td>
    </tr>
    <tr>
      <th>807</th>
      <td>Jackboy - You Can Go (Na Na Na) (Official Video)</td>
      <td>[MUSIC, JACKBOY, YOUCANGO(NANANA)(OFFICIALVIDEO), SNIPERGANG/EMPIRE]</td>
      <td>807</td>
    </tr>
    <tr>
      <th>808</th>
      <td>KISSING MY BEST FRIEND #shorts</td>
      <td>[COMEDY, None, None, None]</td>
      <td>808</td>
    </tr>
    <tr>
      <th>809</th>
      <td>Karens &amp; Cancel Culture w/Chelsea Handler - Uncomfortable Conversations with a Black Man Ep.10</td>
      <td>[PEOPLE&amp;BLOGS, EMMANUELACHO, UNCOMFORTABLECONVERSATION, UNCOMFORTABLECONVERSATIONS]</td>
      <td>809</td>
    </tr>
    <tr>
      <th>810</th>
      <td>Ken Block’s 1,400hp AWD Ford Mustang Hoonicorn vs. a McLaren Senna Merlin // Hoonicorn Vs the World</td>
      <td>[AUTOS&amp;VEHICLES, KENBLOCK, KENBLOCKMUSTANG, KENBLOCKFORDMUSTANG]</td>
      <td>810</td>
    </tr>
    <tr>
      <th>811</th>
      <td>Kodak Black - Every Balmain [Official Audio]</td>
      <td>[MUSIC, KODAK, BLACK, KODAKBLACK]</td>
      <td>811</td>
    </tr>
    <tr>
      <th>812</th>
      <td>MIKE MAJLAK SPEAKS ON BREAKUP WITH LANA RHOADES</td>
      <td>[ENTERTAINMENT, IMPAULSIVECLIPS, IMPAULSIVE, LOGANPAULPODCAST]</td>
      <td>812</td>
    </tr>
    <tr>
      <th>813</th>
      <td>MSNBC cuts away from Trump's address after he again falsely declares election victory</td>
      <td>[NEWS&amp;POLITICS, USELECTION, USELECTION2020, TRUMP]</td>
      <td>813</td>
    </tr>
    <tr>
      <th>814</th>
      <td>Megan Thee Stallion - Don’t Stop (feat. Young Thug) [Official Video]</td>
      <td>[MUSIC, MEGANTHEESTALLION, MEGAN, MEGANTHESTALLION]</td>
      <td>814</td>
    </tr>
    <tr>
      <th>815</th>
      <td>Minecraft but PROS rule the world</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>815</td>
    </tr>
    <tr>
      <th>816</th>
      <td>My Decaying Mind in Quarantine</td>
      <td>[COMEDY, CAFECHAOS, FOODFIGHT, QUARANTINE]</td>
      <td>816</td>
    </tr>
    <tr>
      <th>817</th>
      <td>My Dreams</td>
      <td>[COMEDY, HAMINATIONS, DREAMS, MYDREAMS]</td>
      <td>817</td>
    </tr>
    <tr>
      <th>818</th>
      <td>NLE Choppa - Moonlight feat. Big Sean (Official Music Video)</td>
      <td>[MUSIC, NLECHOPPA, BRYSON, FROMDARKTOLIGHT]</td>
      <td>818</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Naomi Osaka vs Serena Williams Match Highlights (SF) | Australian Open 2021</td>
      <td>[SPORTS, AUSTRALIANOPEN, AUSTRALIANOPEN2021, AUSTRALIANOPEN2021HIGHLIGHTS]</td>
      <td>819</td>
    </tr>
    <tr>
      <th>820</th>
      <td>P!nk - All I Know So Far (Official Video)</td>
      <td>[MUSIC, PINK, P!NK, ALLIKNOWSOFAR]</td>
      <td>820</td>
    </tr>
    <tr>
      <th>821</th>
      <td>POP SMOKE - MOOD SWINGS ft. Lil Tjay (Visualizer)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>821</td>
    </tr>
    <tr>
      <th>822</th>
      <td>Painting My Entire Room With The World's Brightest Paint...Then Turning on a 100,000 Lumen Light!</td>
      <td>[SCIENCE&amp;TECHNOLOGY, MUSOUBLACK, BRIGHTESTPAINT, LIT]</td>
      <td>822</td>
    </tr>
    <tr>
      <th>823</th>
      <td>PnB Rock - Rose Gold (feat. King Von) [Official Music Video]</td>
      <td>[MUSIC, PNBROCK, KINGVON, ROCKVON]</td>
      <td>823</td>
    </tr>
    <tr>
      <th>824</th>
      <td>RC Helicopter Battle | Dude Perfect</td>
      <td>[SPORTS, DUDEPERFECT, DUDEPERFECTSTEREOTYPES, DUDEPERFECTWATERBOTTLEFLIP]</td>
      <td>824</td>
    </tr>
    <tr>
      <th>825</th>
      <td>RETAKE  // Episode 2 Cinematic - VALORANT</td>
      <td>[GAMING, VALORANTEPISODE2, VALORANTCINEMATIC, VALORANT]</td>
      <td>825</td>
    </tr>
    <tr>
      <th>826</th>
      <td>Red Hot Chili Peppers - 2022 Global Stadium Tour Details Revealed</td>
      <td>[MUSIC, REDHOTCHILIPEPPERS, RHCP, ANTHONY]</td>
      <td>826</td>
    </tr>
    <tr>
      <th>827</th>
      <td>Resumen y goles | Japón 0-2 México | Amistoso 2020 | TUDN</td>
      <td>[SPORTS, TELEVISA, UNIVISION, TUDN]</td>
      <td>827</td>
    </tr>
    <tr>
      <th>828</th>
      <td>SWAPPING OUR HOUSE WITH THE LABRANT FAM!</td>
      <td>[PEOPLE&amp;BLOGS, KYLERANDMAD, TAYTUMANDOAKLEY, FISHFAM]</td>
      <td>828</td>
    </tr>
    <tr>
      <th>829</th>
      <td>Snow Tha Product - Never Be Me (Official Music Video)</td>
      <td>[MUSIC, SNOWTHAPRODUCT, MUSICVIDEO, NEVERBEME]</td>
      <td>829</td>
    </tr>
    <tr>
      <th>830</th>
      <td>So.. I'm moving</td>
      <td>[COMEDY, MORGANADAMS, None, None]</td>
      <td>830</td>
    </tr>
    <tr>
      <th>831</th>
      <td>Space Jam: A New Legacy – Trailer 2</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>831</td>
    </tr>
    <tr>
      <th>832</th>
      <td>The Making Of Fallin’ Episode 2 | Why Don’t We</td>
      <td>[MUSIC, WHYDON'TWE, WHYDONTWE, WDW]</td>
      <td>832</td>
    </tr>
    <tr>
      <th>833</th>
      <td>The Paris Hilton you never knew | This Is Paris (Official Trailer)</td>
      <td>[ENTERTAINMENT, PARISHILTON, PARIS, HILTON]</td>
      <td>833</td>
    </tr>
    <tr>
      <th>834</th>
      <td>The Try Guys Bake Pizza Without A Recipe</td>
      <td>[COMEDY, TRYGUYS, KEITH, NED]</td>
      <td>834</td>
    </tr>
    <tr>
      <th>835</th>
      <td>Tom MacDonald - Dont Look Down</td>
      <td>[MUSIC, EMINEM, MGK, TOKEN]</td>
      <td>835</td>
    </tr>
    <tr>
      <th>836</th>
      <td>TotalEnergies BWF Sudirman Cup 2021 | Gideon/Sukamuljo (INA) vs Kolding/Søgaard (DEN) | Group C</td>
      <td>[SPORTS, B_FULL_MATCH, TOTALENERGIESBWFSUDIRMANCUP2021, 2021]</td>
      <td>836</td>
    </tr>
    <tr>
      <th>837</th>
      <td>UFC 257 Embedded: Vlog Series - Episode 2</td>
      <td>[SPORTS, UFC, 257, ABU]</td>
      <td>837</td>
    </tr>
    <tr>
      <th>838</th>
      <td>We Made the World's Largest Candy Hearts • This Could Be Awesome #16</td>
      <td>[ENTERTAINMENT, VAT19, VAT19, VAT-19]</td>
      <td>838</td>
    </tr>
    <tr>
      <th>839</th>
      <td>What If DreamSMP Was In Hardcore Mode?</td>
      <td>[GAMING, None, None, None]</td>
      <td>839</td>
    </tr>
    <tr>
      <th>840</th>
      <td>What TIME does the CROWN TUNDRA RELEASE?! Breakdown</td>
      <td>[GAMING, WHATTIMEDOESTHECROWNTUNDRARELEASE, RELEASETIMECROWNTUNDRA, CROWNTUNDRARELEASEDATE]</td>
      <td>840</td>
    </tr>
    <tr>
      <th>841</th>
      <td>Why Are We Obsessed With People? ft. Jaiden Animations</td>
      <td>[EDUCATION, LIFENOGGIN, EDUCATION, EDUCATIONCHANNEL]</td>
      <td>841</td>
    </tr>
    <tr>
      <th>842</th>
      <td>Why Don't We - Fallin' [Official Music Video]</td>
      <td>[MUSIC, WHYDONTWE, WHYDON'TWE, FALLIN']</td>
      <td>842</td>
    </tr>
    <tr>
      <th>843</th>
      <td>Why I'm Leaving California</td>
      <td>[EDUCATION, INVESTING, INVESTINGFORBEGINNERS, INVESTINGINYOUR20S]</td>
      <td>843</td>
    </tr>
    <tr>
      <th>844</th>
      <td>Young M.A Henny'd Up (Official Music Video)</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>844</td>
    </tr>
    <tr>
      <th>845</th>
      <td>i learnt a new swear word</td>
      <td>[ENTERTAINMENT, SWEARWORD, SWEAR, INSULT]</td>
      <td>845</td>
    </tr>
    <tr>
      <th>846</th>
      <td>microwave my salad and call me a donut | Kitchen Nightmares</td>
      <td>[ENTERTAINMENT, GORDONRAMSAY, KITCHENNIGHTMARES, GORDONRAMSAYKITCHENNIGHTMARES]</td>
      <td>846</td>
    </tr>
    <tr>
      <th>847</th>
      <td>【特報】ユーリ!!! on ICE 劇場版 : ICE ADOLESCENCE（アイス アドレセンス）</td>
      <td>[FILM&amp;ANIMATION, ユーリ!!!ONICE, ユーリ, YURIONICE]</td>
      <td>847</td>
    </tr>
    <tr>
      <th>848</th>
      <td>#PWRUP TEASER 2</td>
      <td>[MUSIC, None, None, None]</td>
      <td>848</td>
    </tr>
    <tr>
      <th>849</th>
      <td>*NEW* RANDOM ROLES *6* in Among Us</td>
      <td>[GAMING, None, None, None]</td>
      <td>849</td>
    </tr>
    <tr>
      <th>850</th>
      <td>Animal Crossing: New Horizons Direct 10.15.2021</td>
      <td>[GAMING, NINTENDO, FUN, ANIMALCROSSING]</td>
      <td>850</td>
    </tr>
    <tr>
      <th>851</th>
      <td>Baryon Mode Is WAY Stronger Than You Think.</td>
      <td>[ENTERTAINMENT, BARYONMODEISSTRONGERTHANYOUTHINK, HOWSTRONGISBARYONMODE, BARYONMODEEXPLAINED]</td>
      <td>851</td>
    </tr>
    <tr>
      <th>852</th>
      <td>Binging with Babish: Meat-Ghetti &amp; Spag-Balls from American Dad</td>
      <td>[PEOPLE&amp;BLOGS, PEARQWERTYHORSE, BINGINGWITHBABISH, BWB]</td>
      <td>852</td>
    </tr>
    <tr>
      <th>853</th>
      <td>Buildings collapse as Israel carries out hundreds of airstrikes in Gaza</td>
      <td>[NEWS&amp;POLITICS, GLOBALNEWS, ISRAELGAZACONFLICT, ISRAELIPALESTINIANCONFLICT]</td>
      <td>853</td>
    </tr>
    <tr>
      <th>854</th>
      <td>Buying The WORST FOUR-WHEELER on FaceBook Market!</td>
      <td>[ENTERTAINMENT, CANAM, DEFENDER, X3]</td>
      <td>854</td>
    </tr>
    <tr>
      <th>855</th>
      <td>California wildfires evening update: September 9, 2020</td>
      <td>[NEWS&amp;POLITICS, WILDFIRE, CALIFORNIA, LOCAL]</td>
      <td>855</td>
    </tr>
    <tr>
      <th>856</th>
      <td>Chris Brown, Young Thug - Say You Love Me (Official Video)</td>
      <td>[MUSIC, YOUNGTHUGFUTURE, UPYOUNGTHUG, FUTURE]</td>
      <td>856</td>
    </tr>
    <tr>
      <th>857</th>
      <td>Chris Harrison Gives Clare Crawley Ultimatum | The Bachelorette</td>
      <td>[ENTERTAINMENT, BACHELORETTE, THEBACHELORETTE, BACHELORETTE2020]</td>
      <td>857</td>
    </tr>
    <tr>
      <th>858</th>
      <td>Christopher Ramirez, the 3-year-old boy missing since Wednesday in Grimes County has been found,...</td>
      <td>[NEWS&amp;POLITICS, LATEST, VIDEOS, None]</td>
      <td>858</td>
    </tr>
    <tr>
      <th>859</th>
      <td>DEBATE: Is Ben Askren In Jake Paul’s Head?</td>
      <td>[SPORTS, TRUEGEORDIE, KSI, JAKEPAUL]</td>
      <td>859</td>
    </tr>
    <tr>
      <th>860</th>
      <td>DOING JAMES CHARLES MAKEUP *WE'RE BACK?*</td>
      <td>[ENTERTAINMENT, EMMACHAMBERLAIN, EMMACHAMBIE, VLOG]</td>
      <td>860</td>
    </tr>
    <tr>
      <th>861</th>
      <td>Destroying my Grandpas Truck and not buying him a new one</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>861</td>
    </tr>
    <tr>
      <th>862</th>
      <td>Earn $20K EVERY MONTH by being your own boss</td>
      <td>[ENTERTAINMENT, BRIAN, DAVID, GILBERT]</td>
      <td>862</td>
    </tr>
    <tr>
      <th>863</th>
      <td>El Salvador vs. USA: Extended Highlights | CONCACAF World Cup Qualifying | CBS Sports Golazo</td>
      <td>[SPORTS, SOCCER, FOOTBALL, UCL]</td>
      <td>863</td>
    </tr>
    <tr>
      <th>864</th>
      <td>FRANCHISE (CACTUS SLATT)</td>
      <td>[MUSIC, TRAVISSCOTT, MUSICVIDEO, CACTUSJACK]</td>
      <td>864</td>
    </tr>
    <tr>
      <th>865</th>
      <td>Fables and Folktales: The Snow Queen</td>
      <td>[EDUCATION, FUNNY, SUMMARY, OSP]</td>
      <td>865</td>
    </tr>
    <tr>
      <th>866</th>
      <td>Film Theory: Follow The Rabbit... Decoding The Walten Files.</td>
      <td>[FILM&amp;ANIMATION, THEWALTENFILES, THEWALTENFILES3, WALTENFILES3]</td>
      <td>866</td>
    </tr>
    <tr>
      <th>867</th>
      <td>French Is Easy pt. 2</td>
      <td>[FILM&amp;ANIMATION, None, None, None]</td>
      <td>867</td>
    </tr>
    <tr>
      <th>868</th>
      <td>HIGHLIGHTS | Rayo Vallecano 1-2 Barça | Copa del Rey</td>
      <td>[SPORTS, FCBARCELONA, برشلونة،, FÚTBOL]</td>
      <td>868</td>
    </tr>
    <tr>
      <th>869</th>
      <td>HotSpanish - 9AM EN MÉXICO (Video Oficial)</td>
      <td>[MUSIC, HOTSPANISH, MUSIC, MUSICA]</td>
      <td>869</td>
    </tr>
    <tr>
      <th>870</th>
      <td>How Do I Always Lose?! Among Us Song (Animated Music Video)</td>
      <td>[FILM&amp;ANIMATION, AMONGUS, AMONGUSLOGIC, AMONGUSSONG]</td>
      <td>870</td>
    </tr>
    <tr>
      <th>871</th>
      <td>How I survived a RANDOMIZED Nuzlocke</td>
      <td>[GAMING, ALPHARAD, SUPERSMASHBROSULTIMATE, SMASHBROS]</td>
      <td>871</td>
    </tr>
    <tr>
      <th>872</th>
      <td>How To Properly Use A Claymore in Rainbow Six Siege😂🤣 #Shorts</td>
      <td>[GAMING, None, None, None]</td>
      <td>872</td>
    </tr>
    <tr>
      <th>873</th>
      <td>I Bought a Cheap REPO Aston Martin at Auction with Mystery Mechanical Damage SIGHT UNSEEN!</td>
      <td>[AUTOS&amp;VEHICLES, ASTONMARTIN, V8, VANTAGE]</td>
      <td>873</td>
    </tr>
    <tr>
      <th>874</th>
      <td>I Bought an AS-IS $90,000 Mercedes AMG at Auction and got 50% OFF (Twin Turbo C63s)</td>
      <td>[AUTOS&amp;VEHICLES, EXHAUST, INSTALL, DIY]</td>
      <td>874</td>
    </tr>
    <tr>
      <th>875</th>
      <td>I OFFICIALLY MOVED TO LA!! (NEW HOUSE TOUR)</td>
      <td>[ENTERTAINMENT, FAZERUG, RUG, RUGFAZE]</td>
      <td>875</td>
    </tr>
    <tr>
      <th>876</th>
      <td>I Tricked FaZe Rug Into Thinking His Car Got Destroyed</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>876</td>
    </tr>
    <tr>
      <th>877</th>
      <td>I Turned My Cat Into Lego!</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>877</td>
    </tr>
    <tr>
      <th>878</th>
      <td>If Your White Friend Was in Squid Game</td>
      <td>[ENTERTAINMENT, FREEREFILLS, FREEREFILLS, TYPESOF]</td>
      <td>878</td>
    </tr>
    <tr>
      <th>879</th>
      <td>Indianapolis shooting: Police ID killer in FedEx shooting as former employee Brandon Scott Hole, 19</td>
      <td>[NEWS&amp;POLITICS, NEWS, FEDEXSHOOTING, INDIANAPOLISSHOOTING]</td>
      <td>879</td>
    </tr>
    <tr>
      <th>880</th>
      <td>Insane Water Slides!</td>
      <td>[ENTERTAINMENT, LIFEHACKS, CRAFTS, SLIME]</td>
      <td>880</td>
    </tr>
    <tr>
      <th>881</th>
      <td>Inside the Lab That Invented the COVID-19 Vaccine</td>
      <td>[EDUCATION, SCIENCE, JOEHANSON, IT'SOKAYTOBESMART]</td>
      <td>881</td>
    </tr>
    <tr>
      <th>882</th>
      <td>Israel launches military offensive on Gaza. I 10 News First</td>
      <td>[NEWS&amp;POLITICS, 10NEWSFIRST, TENNEWSFIRST, None]</td>
      <td>882</td>
    </tr>
    <tr>
      <th>883</th>
      <td>J. Cole - 9 5 . s o u t h (Official Audio)</td>
      <td>[MUSIC, None, None, None]</td>
      <td>883</td>
    </tr>
    <tr>
      <th>884</th>
      <td>JAMIE FOXX GAVE ME A BLACK EYE</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>884</td>
    </tr>
    <tr>
      <th>885</th>
      <td>Joseline Hernandez on Wendy Williams</td>
      <td>[ENTERTAINMENT, WENDYWILLIAMS, THEWENDYWILLIAMSSHOW, #YOUTUBEBLACK]</td>
      <td>885</td>
    </tr>
    <tr>
      <th>886</th>
      <td>Justin Bieber - Hold On (Live with Jason Kennedy)</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>886</td>
    </tr>
    <tr>
      <th>887</th>
      <td>Kodak Black Drops $180K on Jewelry for the Culture!</td>
      <td>[ENTERTAINMENT, ICEBOX, ATLANTA, ATLANTARAP]</td>
      <td>887</td>
    </tr>
    <tr>
      <th>888</th>
      <td>Lil Baby - Errbody (Official Video)</td>
      <td>[MUSIC, LILBABY, ERRBODY, LILBABYERRBODY]</td>
      <td>888</td>
    </tr>
    <tr>
      <th>889</th>
      <td>Lil Reese: Doctors Said My Voice Would Never Come Back, Never Rap Again/ Will It Get Better?</td>
      <td>[ENTERTAINMENT, CAMCAPONENEWS, CAMCAPONENEWS, INTERVIEW]</td>
      <td>889</td>
    </tr>
    <tr>
      <th>890</th>
      <td>MEAN PC Imposter Mod in Among Us</td>
      <td>[GAMING, None, None, None]</td>
      <td>890</td>
    </tr>
    <tr>
      <th>891</th>
      <td>Minecraft secret potion #Shorts</td>
      <td>[COMEDY, MINECRAFT, MINECRAFTANIMATION, MINECRAFTROLEPLAY]</td>
      <td>891</td>
    </tr>
    <tr>
      <th>892</th>
      <td>Miscellaneous Myths: King Midas</td>
      <td>[EDUCATION, FUNNY, SUMMARY, OSP]</td>
      <td>892</td>
    </tr>
    <tr>
      <th>893</th>
      <td>Moneybagg Yo - SRT (feat. BIG30 &amp; Pooh Shiesty) (Official Audio)</td>
      <td>[MUSIC, MONEYBAGG, SRT, (AUDIO)]</td>
      <td>893</td>
    </tr>
    <tr>
      <th>894</th>
      <td>Monster Hunter World: Iceborne - Title Update 5 Trailer</td>
      <td>[ENTERTAINMENT, MONSTERHUNTER, MH, ACTION]</td>
      <td>894</td>
    </tr>
    <tr>
      <th>895</th>
      <td>My Sister Rates My SHEIN Fall Outfits!! Try On Haul</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>895</td>
    </tr>
    <tr>
      <th>896</th>
      <td>New video: What Nearman did after letting protesters in</td>
      <td>[NEWS&amp;POLITICS, DEFAULT, None, None]</td>
      <td>896</td>
    </tr>
    <tr>
      <th>897</th>
      <td>PERFECT TIMING From Level 1 to Level 100</td>
      <td>[SPORTS, THATSAMAZING, THAT'SAMAZING, TRICKSHOTS]</td>
      <td>897</td>
    </tr>
    <tr>
      <th>898</th>
      <td>PRANKING MY FRIENDS WITH MY NEW HAIR!!</td>
      <td>[PEOPLE&amp;BLOGS, PRANKINGMYFRIENDS, PRANKINGMYFRIENDSWITHTHIS, PRANKINGMYFRIENDSWITHMYNEWHAIR]</td>
      <td>898</td>
    </tr>
    <tr>
      <th>899</th>
      <td>PREPPING FOR BABY'S ARRIVAL</td>
      <td>[PEOPLE&amp;BLOGS, ALONDRADESSY, ALOANDBENNY, ALONDRAANDBENNY]</td>
      <td>899</td>
    </tr>
    <tr>
      <th>900</th>
      <td>Planking the Hull - Part 1 (Rebuilding Tally Ho / EP91.1)</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>900</td>
    </tr>
    <tr>
      <th>901</th>
      <td>Polo G - Black Hearted (Official Video)</td>
      <td>[MUSIC, POLOG, BLACKHEARTED, HALLOFFAME]</td>
      <td>901</td>
    </tr>
    <tr>
      <th>902</th>
      <td>Regresa Mami</td>
      <td>[MUSIC, ESLABONARMADOTUVENENOMORTAL, VOL.2REGRESAMAMI, None]</td>
      <td>902</td>
    </tr>
    <tr>
      <th>903</th>
      <td>Rick and Morty | S5E3 Cold Open: Planetina Saves the Day | adult swim</td>
      <td>[ENTERTAINMENT, ADULTSWIM, ADULTANIMATION, COMEDY]</td>
      <td>903</td>
    </tr>
    <tr>
      <th>904</th>
      <td>SM6 - Oddity (Official Music Video)</td>
      <td>[MUSIC, SM6BAND, SM6BAND, SM6]</td>
      <td>904</td>
    </tr>
    <tr>
      <th>905</th>
      <td>Secret Admirer #AEUnderwear Prank with Addison &amp; Bryce | American Eagle</td>
      <td>[COMEDY, #AEXME, AMERICANEAGLE, AMERICANEAGLEJEANS]</td>
      <td>905</td>
    </tr>
    <tr>
      <th>906</th>
      <td>Spending Christmas in the Hospital</td>
      <td>[FILM&amp;ANIMATION, TIMTOM, TIM, TOM]</td>
      <td>906</td>
    </tr>
    <tr>
      <th>907</th>
      <td>Stranger Things 4 | Creel House | Netflix</td>
      <td>[ENTERTAINMENT, CALEBMCLAUGHLIN, CHARLIEHEATON, CHIEFHOPPER]</td>
      <td>907</td>
    </tr>
    <tr>
      <th>908</th>
      <td>Succession (2021) | Season 3 Official Trailer | HBO</td>
      <td>[ENTERTAINMENT, SUCCESSION, HBO, SUCCESSIONHBO]</td>
      <td>908</td>
    </tr>
    <tr>
      <th>909</th>
      <td>Succession: Season 3 | Official Tease | HBO</td>
      <td>[ENTERTAINMENT, SUCCESSION, HBO, HOMEBOXOFFICE]</td>
      <td>909</td>
    </tr>
    <tr>
      <th>910</th>
      <td>Thanks for the last 15 years... Goodbye.</td>
      <td>[ENTERTAINMENT, WASSABI, ALEXWASSABI, WASSABIPRODUCTIONS]</td>
      <td>910</td>
    </tr>
    <tr>
      <th>911</th>
      <td>The Absolute State of Affairs</td>
      <td>[COMEDY, None, None, None]</td>
      <td>911</td>
    </tr>
    <tr>
      <th>912</th>
      <td>The Champions: Season 5, Episode 2</td>
      <td>[SPORTS, BRFOOTBALL, BLEACHERREPORT, SOCCER]</td>
      <td>912</td>
    </tr>
    <tr>
      <th>913</th>
      <td>The Evolution of Shuffling Cards!! - #Shorts</td>
      <td>[ENTERTAINMENT, CARDSHUFFLING, THEEVOLUTIONOFMAGIC, THEEVOLUTIONOFCARDSHUFFLING]</td>
      <td>913</td>
    </tr>
    <tr>
      <th>914</th>
      <td>The Fart Limit #shorts</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>914</td>
    </tr>
    <tr>
      <th>915</th>
      <td>Times NBA Legends DISRESPECTED Eachother..</td>
      <td>[SPORTS, NBA, REBOUNDCENTRAL, REBOUND]</td>
      <td>915</td>
    </tr>
    <tr>
      <th>916</th>
      <td>Toxic Punk</td>
      <td>[MUSIC, None, None, None]</td>
      <td>916</td>
    </tr>
    <tr>
      <th>917</th>
      <td>WE CAN’T BELIEVE DABABY DID THIS AT COOLKICKS</td>
      <td>[ENTERTAINMENT, COOLKICKS, COOLKICKS, COOLKICKSYOUTUBE]</td>
      <td>917</td>
    </tr>
    <tr>
      <th>918</th>
      <td>WE OPEN A STARBUCKS IN BRENT'S NEW HOUSE!!</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>918</td>
    </tr>
    <tr>
      <th>919</th>
      <td>Warframe | TennoCon 2021 Teaser</td>
      <td>[GAMING, WARFRAME, TENNOCON, TENNOCON2021]</td>
      <td>919</td>
    </tr>
    <tr>
      <th>920</th>
      <td>Who Hacked Apex Legends and Why , Fixed Now</td>
      <td>[GAMING, APEXLEGENDSHACKEDTODAY, APEXLEGENDSTITANFALLHACK, APEXLEGENDSHOWTOGETINGAME]</td>
      <td>920</td>
    </tr>
    <tr>
      <th>921</th>
      <td>Without Remorse - Final Trailer | Prime Video</td>
      <td>[ENTERTAINMENT, WITHOUTREMORSETRAILER, WITHOUTREMORSETRAILER2021, WITHOUTREMORSEMICHAELBJORDAN]</td>
      <td>921</td>
    </tr>
    <tr>
      <th>922</th>
      <td>World's Largest Water Tunnel!</td>
      <td>[ENTERTAINMENT, LIFEHACKS, CRAFTS, SLIME]</td>
      <td>922</td>
    </tr>
    <tr>
      <th>923</th>
      <td>Wow..We're like..Parents Now...</td>
      <td>[PEOPLE&amp;BLOGS, SLICENRICE, SLICEANDRICE, SLICENRICEBABY]</td>
      <td>923</td>
    </tr>
    <tr>
      <th>924</th>
      <td>You Laugh You Lose (HOT WINGS EDITION)</td>
      <td>[PEOPLE&amp;BLOGS, SHORTS, THEBOYSSHORTS, VR]</td>
      <td>924</td>
    </tr>
    <tr>
      <th>925</th>
      <td>[FNF] Making Black Impostor V3 Sculpture Timelapse [Among us] - Friday Night Funkin' Mods</td>
      <td>[GAMING, FIGURE, 1:6SCALE, 피규어]</td>
      <td>925</td>
    </tr>
    <tr>
      <th>926</th>
      <td>[MV] 마마무 (MAMAMOO) - 딩가딩가 (Dingga)</td>
      <td>[MUSIC, MAMAMOO, 마마무, 딩가딩가]</td>
      <td>926</td>
    </tr>
    <tr>
      <th>927</th>
      <td>see you soon</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>927</td>
    </tr>
    <tr>
      <th>928</th>
      <td>the TRUTH about Canadian peanut butter #shorts</td>
      <td>[EDUCATION, None, None, None]</td>
      <td>928</td>
    </tr>
    <tr>
      <th>929</th>
      <td>the guy who did 1 night in jail</td>
      <td>[COMEDY, TREVORWALLACE, TREVORWALACE, TRAVISWALLACE]</td>
      <td>929</td>
    </tr>
    <tr>
      <th>930</th>
      <td>2 Chainz - Can't Go For That ft. Ty Dolla $ign, Lil Duval</td>
      <td>[MUSIC, CHAINZ, CAN'T, FOR]</td>
      <td>930</td>
    </tr>
    <tr>
      <th>931</th>
      <td>ATEEZ(에이티즈) - ‘Eternal Sunshine’ Official MV</td>
      <td>[MUSIC, KQ, 케이큐, 에이티즈]</td>
      <td>931</td>
    </tr>
    <tr>
      <th>932</th>
      <td>AmongKeyboard Keyboard Application for Among Us #Shorts #AmongUs</td>
      <td>[GAMING, AMONGUS, AMONGLOCK, AMONGUSLOCK]</td>
      <td>932</td>
    </tr>
    <tr>
      <th>933</th>
      <td>BLACKPINK - 'Ice Cream (with Selena Gomez)' M/V</td>
      <td>[MUSIC, YGENTERTAINMENT, YG, 와이지]</td>
      <td>933</td>
    </tr>
    <tr>
      <th>934</th>
      <td>Best Sprinkle Art Wins $5,000 Challenge! | ZHC Crafts</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>934</td>
    </tr>
    <tr>
      <th>935</th>
      <td>Blowing Up Earth!</td>
      <td>[GAMING, None, None, None]</td>
      <td>935</td>
    </tr>
    <tr>
      <th>936</th>
      <td>Bob Ross in a Random Outfit</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>936</td>
    </tr>
    <tr>
      <th>937</th>
      <td>CHEAPEST People On The Internet</td>
      <td>[ENTERTAINMENT, SSSNIPERWOLF, SNIPERWOLF, REACTING]</td>
      <td>937</td>
    </tr>
    <tr>
      <th>938</th>
      <td>CORRUPTED (S2 P2) SARV &amp; RUV ~Friday Night Funkin~ [ANIMATION]</td>
      <td>[ENTERTAINMENT, FRIDAYNIGHTFUNKIN, FNF, FNFANIMATION]</td>
      <td>938</td>
    </tr>
    <tr>
      <th>939</th>
      <td>Cochise - Tell Em ft. $NOT (Directed by Cole Bennett)</td>
      <td>[ENTERTAINMENT, COCHISE, TELLEM, WHATSUP]</td>
      <td>939</td>
    </tr>
    <tr>
      <th>940</th>
      <td>Customizing A Mansion In 50 Hours ft. Steve Aoki | ZHC Crafts</td>
      <td>[HOWTO&amp;STYLE, None, None, None]</td>
      <td>940</td>
    </tr>
    <tr>
      <th>941</th>
      <td>DIGTOK (w/ Drew Gooden and Kurtis Conner)</td>
      <td>[COMEDY, DANNYGONZALEZ, FUNNY, COMMENTARY]</td>
      <td>941</td>
    </tr>
    <tr>
      <th>942</th>
      <td>Daycare Stories</td>
      <td>[FILM&amp;ANIMATION, None, None, None]</td>
      <td>942</td>
    </tr>
    <tr>
      <th>943</th>
      <td>Dream - Minecraft Hitmen Extra Scenes</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>943</td>
    </tr>
    <tr>
      <th>944</th>
      <td>El Privilegio de Amar - Mijares feat Lucero Mijares (Sinfonico Online)</td>
      <td>[MUSIC, MIJARESSINFÓNICO, MANUELMIJARES, MIJARES]</td>
      <td>944</td>
    </tr>
    <tr>
      <th>945</th>
      <td>Everything New In Fortnite Chapter 2 Season 6! - Battle Pass, Map, Weapons &amp; More!</td>
      <td>[GAMING, FORTNITESEASON6, EVERYTHINGNEWINFORTNITESEASON6, CHAPTER2SEASON6]</td>
      <td>945</td>
    </tr>
    <tr>
      <th>946</th>
      <td>Extended Highlights: USA 1-0 Mexico - 2021 Gold Cup Final</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>946</td>
    </tr>
    <tr>
      <th>947</th>
      <td>FIGHTING TIKTOKERS ON ROBLOX</td>
      <td>[PEOPLE&amp;BLOGS, LARRAY, ROBLOX, LARRAYROBLOX]</td>
      <td>947</td>
    </tr>
    <tr>
      <th>948</th>
      <td>Film Theory: The Lorax Movie LIED To You!</td>
      <td>[FILM&amp;ANIMATION, THELORAX, LORAX, THELORAXMOVIE]</td>
      <td>948</td>
    </tr>
    <tr>
      <th>949</th>
      <td>Fortnite Update 15.40: EVERYTHING You Need To Know In UNDER 5 MINUTES (Flintknock, Midas, &amp; More!)</td>
      <td>[GAMING, PROGUIDES, FORTNITEPROGUIDES, HOWTOIMPROVEINFORTNITE]</td>
      <td>949</td>
    </tr>
    <tr>
      <th>950</th>
      <td>Fran Rozzano - Inédito [Official Video]</td>
      <td>[MUSIC, RAPHYPINA, RAFAELPINA, PLANB]</td>
      <td>950</td>
    </tr>
    <tr>
      <th>951</th>
      <td>God Of War Ragnarok - PlayStation Showcase 2021 Reveal Trailer | PS5</td>
      <td>[GAMING, GODOFWAR, RAGNAROK, REVEAL]</td>
      <td>951</td>
    </tr>
    <tr>
      <th>952</th>
      <td>HEAT at CELTICS | FULL GAME HIGHLIGHTS | September 15, 2020</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>952</td>
    </tr>
    <tr>
      <th>953</th>
      <td>Hairdresser Reacts To DIY Y2k Stripe Highlights</td>
      <td>[ENTERTAINMENT, HAIRDRESSERREACTSTODIYY2KSTRIPEHIGHLIGHTS, BRADMONDO, BRADMONDONYC]</td>
      <td>953</td>
    </tr>
    <tr>
      <th>954</th>
      <td>Holo Taco UNICORN DREAM Collection Reveal🦄☁️</td>
      <td>[ENTERTAINMENT, NAILS, NAILART, NAILTUTORIAL]</td>
      <td>954</td>
    </tr>
    <tr>
      <th>955</th>
      <td>I AM IN PAIN!</td>
      <td>[GAMING, JACKSEPTICEYE, IAMFISH, IAMFISHGAME]</td>
      <td>955</td>
    </tr>
    <tr>
      <th>956</th>
      <td>I am finally ready to speak out.. My truth..</td>
      <td>[PEOPLE&amp;BLOGS, OHTRETRETREIAMFINALLYREADYTOSPEAKOUT..MYTRUTH.., IAMFINALLYREADYTOSPEAKOUT..MYTRUTH.., ALONDRADESSYWEBROKEUP.]</td>
      <td>956</td>
    </tr>
    <tr>
      <th>957</th>
      <td>I spent a day with LEGENDARY OG MINECRAFTERS (DanTDM, SkyDoesMinecraft, LDShadowLady)</td>
      <td>[EDUCATION, ANTHONYPADILLA, PADILLA, ANTHONY]</td>
      <td>957</td>
    </tr>
    <tr>
      <th>958</th>
      <td>Into the crater!</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>958</td>
    </tr>
    <tr>
      <th>959</th>
      <td>JINJER - Vortex (Official Video) | Napalm Records</td>
      <td>[MUSIC, NAPALMRECORDS, None, None]</td>
      <td>959</td>
    </tr>
    <tr>
      <th>960</th>
      <td>Jake Paul reacts to Nate Robinson KO; Promises Conor McGregor fight will happen</td>
      <td>[SPORTS, CONORMCGREGOR, UFC, DANAWHITE]</td>
      <td>960</td>
    </tr>
    <tr>
      <th>961</th>
      <td>Joyner Lucas - Zim Zimma (starring Mark Wahlberg, George Lopez &amp; Diddy)</td>
      <td>[MUSIC, ZIMZIMMA, JOYNERLUCAS, JOYNERLUCASZIMZIMMA]</td>
      <td>961</td>
    </tr>
    <tr>
      <th>962</th>
      <td>Lil Tecca - Never Left (Official Teaser)</td>
      <td>[MUSIC, KNOWITSGONBEEASYFORME, TECCA, TECA]</td>
      <td>962</td>
    </tr>
    <tr>
      <th>963</th>
      <td>Listen Closely, I got something to tell you!</td>
      <td>[ENTERTAINMENT, THEREALHOUSEWIVESOFATLANTA, NENELEAKES, CYNTHIABAILEY]</td>
      <td>963</td>
    </tr>
    <tr>
      <th>964</th>
      <td>MAGIC at BUCKS | FULL GAME HIGHLIGHTS | August 18, 2020</td>
      <td>[SPORTS, NBA, GLEAGUE, BASKETBALL]</td>
      <td>964</td>
    </tr>
    <tr>
      <th>965</th>
      <td>Meredith Gives Richard Her Power of Attorney - Grey's Anatomy</td>
      <td>[ENTERTAINMENT, HOSPITAL, PATIENTS, EMERGENCY]</td>
      <td>965</td>
    </tr>
    <tr>
      <th>966</th>
      <td>Minecraft Manhunt, But Clutching Gives OP Items FINALE</td>
      <td>[GAMING, MINECRAFT, CHALLENGE, MINECRAFTCHALLENGE]</td>
      <td>966</td>
    </tr>
    <tr>
      <th>967</th>
      <td>Minecraft, But Crafting Is Reversed...</td>
      <td>[GAMING, MINECRAFT, MINECRAFTBUT, MCBUT]</td>
      <td>967</td>
    </tr>
    <tr>
      <th>968</th>
      <td>Moneybagg Yo - Free Promo (feat. Polo G &amp; Lil Durk) (Official Video)</td>
      <td>[MUSIC, MONEYBAGGYO, MONEYBAGYO, AGANGSTA’SPAIN]</td>
      <td>968</td>
    </tr>
    <tr>
      <th>969</th>
      <td>My Coronavirus Update</td>
      <td>[COMEDY, CHRISKLEMENS, CHRISKLEMENS, KLEMENS]</td>
      <td>969</td>
    </tr>
    <tr>
      <th>970</th>
      <td>My NEW House Tour | De'arra Taylor</td>
      <td>[PEOPLE&amp;BLOGS, DEARRATAYLOR, DEARRA, DK4LCHALLENGES]</td>
      <td>970</td>
    </tr>
    <tr>
      <th>971</th>
      <td>Mönchengladbach vs. Real Madrid: Extended Highlights | UCL on CBS Sports</td>
      <td>[SPORTS, REALMADRID, BORUSSIAMÖNCHENGLADBACH, BORUSSIAMÖNCHENGLADBACHVS.REALMADRID]</td>
      <td>971</td>
    </tr>
    <tr>
      <th>972</th>
      <td>NEW Lucas the Spider episodes Coming!</td>
      <td>[ENTERTAINMENT, None, None, None]</td>
      <td>972</td>
    </tr>
    <tr>
      <th>973</th>
      <td>Nick Diaz UFC 266 Press Conference Highlights and Face-off</td>
      <td>[SPORTS, UFC, MMA, UFC266]</td>
      <td>973</td>
    </tr>
    <tr>
      <th>974</th>
      <td>People that used to play instruments during war.</td>
      <td>[COMEDY, None, None, None]</td>
      <td>974</td>
    </tr>
    <tr>
      <th>975</th>
      <td>Pro Fighters react to Sean O'Malley's injury &amp; loss to Marlon 'Chito' Vera at UFC 252</td>
      <td>[SPORTS, ESPNMMA, MMANEWS, UFC:ULTIMATEFIGHTINGCHAMPIONSHIPS]</td>
      <td>975</td>
    </tr>
    <tr>
      <th>976</th>
      <td>Producer Reacts to Olivia Rodrigo - good 4 u</td>
      <td>[MUSIC, BLAKEMCLAIN, BLAKEMCLAINREACTS, PRODUCERREACTS]</td>
      <td>976</td>
    </tr>
    <tr>
      <th>977</th>
      <td>Resumen de Athletic Club vs FC Barcelona (1-1)</td>
      <td>[SPORTS, LIGA, LALIGA, LALIGASANTANDER]</td>
      <td>977</td>
    </tr>
    <tr>
      <th>978</th>
      <td>Rogue Queen Takes It All (Clash Of Clans Season Challenges)</td>
      <td>[GAMING, CLASHOFCLANS, COC, CLASHOFCLANSGAMEPLAY]</td>
      <td>978</td>
    </tr>
    <tr>
      <th>979</th>
      <td>SHOW DE NEYMAR! BRASIL VENCE O PERU NAS ELIMINATÓRIAS DA COPA - MELHORES MOMENTOS (13/10/2020)</td>
      <td>[SPORTS, ESPORTEINTERATIVO, BRASIL, PERU]</td>
      <td>979</td>
    </tr>
    <tr>
      <th>980</th>
      <td>Saying Goodbye to our Cabin (life update)</td>
      <td>[TRAVEL&amp;EVENTS, CABINLIFE, CABIN, EAMONANDBEC]</td>
      <td>980</td>
    </tr>
    <tr>
      <th>981</th>
      <td>Stephen A.: The 76ers should trade Ben Simmons to the Trail Blazers | First Take</td>
      <td>[SPORTS, FIRSTTAKE, ESPN, NBA]</td>
      <td>981</td>
    </tr>
    <tr>
      <th>982</th>
      <td>Suni Lee’s Journey to Bringing Home the Gold | Golden: The Journey of USA's Elite Gymnasts | Peacock</td>
      <td>[ENTERTAINMENT, PEACOCK, PEACOCKTV, PEACOCKSTREAMINGSERVICE]</td>
      <td>982</td>
    </tr>
    <tr>
      <th>983</th>
      <td>Sus Bus!!! | Among Us Animation</td>
      <td>[FILM&amp;ANIMATION, SWOOZIE, ADANDE, ANIMATION]</td>
      <td>983</td>
    </tr>
    <tr>
      <th>984</th>
      <td>Suspected Triggerman in Shooting of Two Deputies Faces Charges | NBCLA</td>
      <td>[NEWS&amp;POLITICS, NEWS, LOSANGELES, NBCNEWS]</td>
      <td>984</td>
    </tr>
    <tr>
      <th>985</th>
      <td>TRICKY PHASE 3/4 IS HERE AND I'M MAD - Friday Night Funkin' Vs Tricky Mod V2 Showcase/Reaction</td>
      <td>[GAMING, TRICKYMOD, TRICKYPHASE3, TRICKYPHASE4]</td>
      <td>985</td>
    </tr>
    <tr>
      <th>986</th>
      <td>Tempo - Loco Los Dejo (Video Oficial)</td>
      <td>[MUSIC, TEMPO, LOCOLOSDEJO, LOCOLOSDEJOTEMPO]</td>
      <td>986</td>
    </tr>
    <tr>
      <th>987</th>
      <td>Terence Crawford Highlight Reel Knockout of Kell Brook, Pacquiao Next | FULL FIGHT HIGHLIGHTS</td>
      <td>[SPORTS, TERENCECRAWFORD, TERENCECRAWFORDHIGHLIGHTS, TERENCECRAWFORDKNOCKOUTS]</td>
      <td>987</td>
    </tr>
    <tr>
      <th>988</th>
      <td>The BTS Meal | McDonald’s</td>
      <td>[FILM&amp;ANIMATION, BTS, MCDONALD'S, BTSMEAL]</td>
      <td>988</td>
    </tr>
    <tr>
      <th>989</th>
      <td>The Demonic Possession of the Conjuring House</td>
      <td>[PEOPLE&amp;BLOGS, BUZZFEED, BUZZFEEDUNSOLVED, BUZZFEEDUNSOLVED]</td>
      <td>989</td>
    </tr>
    <tr>
      <th>990</th>
      <td>The Green Knight | Official Trailer HD | A24</td>
      <td>[FILM&amp;ANIMATION, A24, A24FILMS, A24TRAILERS]</td>
      <td>990</td>
    </tr>
    <tr>
      <th>991</th>
      <td>UFC 257: Conor McGregor vs Dustin Poirier Recap</td>
      <td>[SPORTS, MMA, MIXEDMARTIALARTS, MMAWEEKLY.COM]</td>
      <td>991</td>
    </tr>
    <tr>
      <th>992</th>
      <td>VALORANT | New Agent CHAMBER - Abilities Explained In 2 Minutes</td>
      <td>[GAMING, HITSCAN, MYSCA, RYANCENTRAL]</td>
      <td>992</td>
    </tr>
    <tr>
      <th>993</th>
      <td>VECINITA 3 Remix - Frankely MC ✖️ Lil Rosse ✖️ Jc La Nevula ✖️ La Ross Maria (Video Oficial)</td>
      <td>[MUSIC, FRANKELYMC, LAROSSMARIA, VECINITA]</td>
      <td>993</td>
    </tr>
    <tr>
      <th>994</th>
      <td>WE FLIRTED AND HE GOT MAD!</td>
      <td>[ENTERTAINMENT, SOFIEDOSSI, AMERICA'SGOTTALENT2016, AGT]</td>
      <td>994</td>
    </tr>
    <tr>
      <th>995</th>
      <td>WE LOST OUR BABY 💔</td>
      <td>[PEOPLE&amp;BLOGS, CARMENANDCOREY, WELOSTOURBABY, PREGNANT]</td>
      <td>995</td>
    </tr>
    <tr>
      <th>996</th>
      <td>WE SURPRISED FAZE RUG WITH THIS ...</td>
      <td>[ENTERTAINMENT, MAMARUG, PAPARUG, MAMARUGANDPAPARUG]</td>
      <td>996</td>
    </tr>
    <tr>
      <th>997</th>
      <td>WE'RE PREGNANT!</td>
      <td>[PEOPLE&amp;BLOGS, None, None, None]</td>
      <td>997</td>
    </tr>
    <tr>
      <th>998</th>
      <td>Walking Across The Entire Universe!</td>
      <td>[GAMING, None, None, None]</td>
      <td>998</td>
    </tr>
    <tr>
      <th>999</th>
      <td>Watch Lil Baby’s birthday celebration on RELEASED</td>
      <td>[MUSIC, NEWMUSIC, NEWMUSICFRIDAY'S, RELEASEDYOUTUBEORIGINALS]</td>
      <td>999</td>
    </tr>
  </tbody>
</table>
</div>
```
:::

::: {.output .stream .stdout}
    time: 434 ms (started: 2021-12-10 18:16:07 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":335,\"base_uri\":\"https://localhost:8080/\"}" id="8u5exMPI3Zkw" outputId="61637223-1698-4d86-e268-1945f5d36d0e"}
``` {.python}
pd.set_option('display.max_rows', 1000)
pandasDFRecommendation.rename(columns = {'combined_small':'Category_Tags'})[['Search Title','Category_Tags']].unique()
```

::: {.output .error ename="AttributeError" evalue="ignored"}
    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    <ipython-input-105-a51bf61467ce> in <module>()
          1 pd.set_option('display.max_rows', 1000)
    ----> 2 pandasDFRecommendation.rename(columns = {'combined_small':'Category_Tags'})[['Search Title','Category_Tags']].unique()

    /usr/local/lib/python3.7/dist-packages/pandas/core/generic.py in __getattr__(self, name)
       5139             if self._info_axis._can_hold_identifiers_and_holds_name(name):
       5140                 return self[name]
    -> 5141             return object.__getattribute__(self, name)
       5142 
       5143     def __setattr__(self, name: str, value) -> None:

    AttributeError: 'DataFrame' object has no attribute 'unique'
:::

::: {.output .stream .stdout}
    time: 40.3 ms (started: 2021-12-10 18:13:26 +00:00)
:::
:::

::: {.cell .code colab="{\"height\":527,\"base_uri\":\"https://localhost:8080/\"}" id="sPH7CNFnpVSF" outputId="e4a785d7-3547-43bc-d7d9-c578fc6de6c6"}
``` {.python}
import pandas as pd
pd.set_option('display.max_colwidth', None)
def getRecommendation(video):
  print("\n\n\n")
  print("Search For Recommendation for Video", video, "\n\n\n")
  #displaylist = pandasDFRecommendation.loc[pandasDFRecommendation['Search Title'] == video].sort_values(by = 'dot', ascending=False)['Recommendation Title', 'combined_small'].head(10).tolist()
  #for i in displaylist:
  #  print(i)
  #print(displaydf)
  return pandasDFRecommendation.loc[pandasDFRecommendation['Search Title'] == video].sort_values(by = 'dot', ascending=True).head(10)

dfrecommend = getRecommendation("REVIVED - Derivakat [Dream SMP original song]")
dfrecommend.reset_index(drop=True, inplace=True)
dfrecommend.rename(columns = {'combined_small':'Category_Tags'}, inplace = True)
dfrecommend.index = dfrecommend.index + 1
dfrecommend[["Recommendation Title", "Category_Tags"]]
```

::: {.output .stream .stdout}




    Search For Recommendation for Video REVIVED - Derivakat [Dream SMP original song] 
:::

::: {.output .execute_result execution_count="101"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recommendation Title</th>
      <th>Category_Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Burna Boy - Monsters You Made [Official Music Video]</td>
      <td>[MUSIC, BURNABOY, BURNERBOY, BURNABOY]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BlocBoy JB - FatBoy (Intro) [Official Music Video]</td>
      <td>[MUSIC, BLOCBOY, BLOCBOYJB, BLOCBOY]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SpotemGottem - Sosa Flow (Official Video)</td>
      <td>[MUSIC, SPOTEMGOTTEM, SPOTEMGOTEM, SPOTEMGOTTEM]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SM6 - Oddity (Official Music Video)</td>
      <td>[MUSIC, SM6BAND, SM6BAND, SM6]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>YUNGBLUD with Denzel Curry - Lemonade</td>
      <td>[MUSIC, YUNGBLUD, YUNGBLUD, YOUNGBLOOD]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Big Scarr Covers Gucci Mane's Hit Song Big Boy Diamonds I 17 Bars</td>
      <td>[MUSIC, AUDIOMACK, AUDIOMACK, TRAPSYMPHONY]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>42 Dugg - Maybach feat. Future (Official Music Video)</td>
      <td>[MUSIC, 42DUGG, 42DUGG, DUGG]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jennifer Lopez - This Land Is Your Land &amp; America, The Beautiful - Inauguration 2021 Performance</td>
      <td>[MUSIC, JENNIFERLOPEZ, JLO, JLO]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Adele - Easy On Me (Clip)</td>
      <td>[MUSIC, ADELE, EASYONME, EOM]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ashton Irwin - Skinny Skinny (Official Music Video)</td>
      <td>[MUSIC, ASHTONIRWIN, SKINNYSKINNY, OFFICIALMUSICVIDEO]</td>
    </tr>
  </tbody>
</table>
</div>
```
:::

::: {.output .stream .stdout}
    time: 52.6 ms (started: 2021-12-10 18:03:37 +00:00)
:::
:::
