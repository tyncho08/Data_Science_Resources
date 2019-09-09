## Lists
- [Apache Foundation](https://www.apache.org/)
    + [Apache Projects List (by category)](https://projects.apache.org/projects.html?category)
    
## Big Data and Data Pipelines
- Application Architecture, Configuration, and Deployment
    + [Apache Brooklyn](https://brooklyn.apache.org/documentation/index.html) - A framework for modeling, monitoring, and managing applications through autonomic blueprints
    + [Apache Bigtop](http://bigtop.apache.org/) - Project for Infrastructure Engineers and Data Scientists looking for comprehensive packaging, testing, and configuration of the leading open source big data components
    + [Apache REEF](http://reef.apache.org/introduction.html) - Apache REEF (Retainable Evaluator Execution Framework) is a library for developing portable applications for cluster resource managers such as Apache Hadoop YARN or Apache Mesos
    + [Apache Slider](https://slider.incubator.apache.org/) - An application to deploy existing distributed applications on an Apache Hadoop YARN cluster, monitor them and make them larger or smaller as desired -even while the application is running
- Data Storage, Resource Management, and Architecture
    + [Apache Hadoop](http://hadoop.apache.org/docs/current/) - Open-source software for reliable, scalable, distributed computing
    + [Apache Hadoop HDFS](http://hadoop.apache.org/docs/r2.7.2/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html) - The primary distributed storage used by Hadoop applications
    + [Apache Hadoop YARN](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html) - Pluggable architecture and resource management for data processing engines to interact with data stored in HDFS
    - Enterprise and Service-based Hadoop
        * [Cloudera](http://www.cloudera.com/documentation.html)
            - [Hortonworks Data Platform (HDP)](http://hortonworks.com/products/data-center/hdp/)
        * [Hortonworks](http://maprdocs.mapr.com/home/)
        * [MapR](http://docs.hortonworks.com/index.html)
            - [MapR Converged Data Platform](MapR Converged Data Platform)
    + [Kite](http://kitesdk.org/docs/current/) - A high-level data layer for Hadoop
- Data Access
    + [Apache Pig](http://pig.apache.org/docs/r0.16.0/) - A platform for analyzing large data sets that consists of a high-level language for expressing data analysis `programs, coupled with infrastructure for evaluating these programs
    + [Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual) - Data warehouse software that facilitates reading, writing, and managing large datasets residing in distributed storage using SQL
    + [Apache Tez](https://tez.apache.org/user_guides.html) - An application framework which allows for a complex directed-acyclic-graph of tasks for processing data
    + [Apache HBase](https://hbase.apache.org/book.html) - Apache HBase is the Hadoop database, a distributed, scalable, big data store
    + [Apache Kudu](https://kudu.apache.org/docs/) - Completes Hadoop's storage layer to enable fast analytics on fast data
    + [Cloudera Impala](http://www.cloudera.com/products/apache-hadoop/impala.html) - The open source, analytic MPP database for Apache Hadoop that provides the fastest time-to-insight
    + [Apache Hive HCatalog](https://cwiki.apache.org/confluence/display/Hive/HCatalog) - A table and storage management layer for Hadoop that enables users with different data processing tools — Pig, MapReduce — to more easily read and write data on the grid
    + [Apache Drill](https://drill.apache.org/docs/) - Schema-free SQL Query Engine for Hadoop, NoSQL and Cloud Storage
    + [Presto](https://prestodb.io/docs/current/) - An open source distributed SQL query engine for running interactive analytic queries against data sources of all sizes ranging from gigabytes to petabytes
- Data Cleaning and Integrity
    + [OpenRefine](http://openrefine.org/documentation.html) - A powerful tool for working with messy data: cleaning it; transforming it from one format into another; and extending it with web services and external data
    + [DataCleaner](http://datacleaner.org/docs) - A strong data profiling engine for discovering and analyzing the quality of your data
- Data Ingestion and Integration
    + [Apache Kafka](https://kafka.apache.org/) - A distributed streaming platform
    + [Apache Spark Streaming](http://spark.apache.org/docs/latest/streaming-programming-guide.html) - An extension of the core Spark API that enables scalable, high-throughput, fault-tolerant stream processing of live data streams
    + [Apache Sqoop](http://sqoop.apache.org/) - A tool designed for efficiently transferring bulk data between Apache Hadoop and structured datastores such as relational databases
    + [Apache Flume](https://flume.apache.org/documentation.html) - A distributed, reliable, and available service for efficiently collecting, aggregating, and moving large amounts of log data
    + [Apache Storm](http://storm.apache.org/index.html) - A free and open source distributed realtime computation system
    + [Talend Open Studio](https://www.talend.com/products/talend-open-studio) - Open source integration software provider to data-driven enterprises
    + [Pentaho Kettle](http://community.pentaho.com/projects/data-integration/)
    + [Blockspring](https://www.blockspring.com/)
    + [Apache Falcon](https://falcon.apache.org/index.html) - A feed processing and feed management system aimed at making it easier for end consumers to onboard their feed processing and feed management on hadoop clusters
    + [Logstash](https://falcon.apache.org/index.html) - Logstash is an open source, server-side data processing pipeline that ingests data from a multitude of sources simultaneously, transforms it, and then sends it to your favorite “stash.”
- ETL/ELT
    + [Stitch](https://www.stitchdata.com/)
    + [Fivetran](https://fivetran.com/)
    + [Singer](https://www.singer.io/) - Simple, Composable, Open Source ETL
- Data Processing (Batch, real-time/streaming, ...)
    + [Apache Spark](http://spark.apache.org/docs/latest/) - A fast and general engine for large-scale data processing
        * [Spark SQL](http://spark.apache.org/docs/latest/sql-programming-guide.html) - A Spark module for structured data processing
        * [MLlib](http://spark.apache.org/docs/latest/ml-guide.html) - Spark’s machine learning (ML) library
        * [GraphX](http://spark.apache.org/docs/latest/graphx-programming-guide.html) - Graphs and graph-parallel computation
        * [Spark Streaming](http://spark.apache.org/docs/latest/streaming-programming-guide.html) - An extension of the core Spark API that enables scalable, high-throughput, fault-tolerant stream processing of live data streams
    + [AWS Kinesis](https://aws.amazon.com/documentation/kinesis/) - Real-time streaming data in the AWS cloud
        * Firehouse - Easily load real-time streaming data into AWS
        * Analytics - Get actionable insights from streaming data in real-time
        * Streams - Build custom applications that process or analyze streaming data for specialized needs
    + [Apache Hadoop MapReduce](https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html) - A YARN-based system for parallel processing of large data sets
    + [Apache Apex](https://apex.apache.org/docs.html) - Enterprise-grade unified stream and batch processing engine
    + [Apache Samza](http://samza.apache.org/learn/documentation/0.11/) - A distributed stream processing framework
    + [Apache Storm](http://storm.apache.org/index.html) - A free and open source distributed realtime computation system
    + [Apache Ignite](https://apacheignite.readme.io/docs) - A high-performance, integrated and distributed in-memory platform for computing and transacting on large-scale data sets in real-time, orders of magnitude faster than possible with traditional disk-based or flash technologies
    + [Apache Flink](https://flink.apache.org/) - A streaming dataflow engine that provides data distribution, communication, and fault tolerance for distributed computations over data streams
    + [Google Cloud Dataflow](https://cloud.google.com/dataflow/docs/) - A fully-managed cloud service and programming model for batch and streaming big data processing
    + [AWS Data Pipeline](https://aws.amazon.com/documentation/data-pipeline/) - Easily automate the movement and transformation of data
    + [AWS EMR](https://aws.amazon.com/documentation/elastic-mapreduce/) - Easily Run and Scale Apache Hadoop, Spark, HBase, Presto, Hive, and other Big Data Frameworks
    + [Google Cloud Dataproc](https://cloud.google.com/dataproc/docs/) - A managed Apache Spark and Apache Hadoop service that lets you take advantage of open source data tools for batch processing, querying, streaming, and machine learning
- Serverless, real-time analytics
    + [AirBnB's StreamAlert](https://github.com/airbnb/streamalert?imm_mid=0ed682&cmp=em-data-na-na-newsltr_20170215)
- Scheduling and Jobs
    + [Apache Oozie](http://oozie.apache.org/) - A workflow scheduler system to manage Apache Hadoop jobs
    + [Spotify's Luigi](https://github.com/spotify/luigi) - Python module that helps you build complex pipelines of batch jobs.
    + [LinkedIn's Azkaban](https://azkaban.github.io/) - Open-source Workflow Manager
    + [Apache Airflow (incubating)](https://github.com/apache/incubator-airflow) - A platform to programmatically author, schedule and monitor workflows
- Analytics
    + [Apache Mahout](http://mahout.apache.org/developers/developer-resources.html) - Build an environment for quickly creating scalable performant machine learning applications
    + [Apache SystemML](https://apache.github.io/incubator-systemml/) - A machine learning platform optimal for big data
    + [Apache Kylin](http://kylin.apache.org/docs15/) - An open source Distributed Analytics Engine designed to provide SQL interface and multi-dimensional analysis (OLAP) on Hadoop supporting extremely large datasets, original contributed from eBay Inc
    + [Apache Lens](http://lens.apache.org/user/index.html) - A unified analytics interface
    + [Programming with Big Data in R" project (pbdR)](https://rbigdata.github.io/index.html)
- Hadoop Enabled Applications
    + [Cascading](http://www.cascading.org/documentation/) - The proven application development platform for building data applications on Hadoop
    + [Cascalog](http://nathanmarz.github.io/cascalog/) - Fully-featured data processing and querying library 
for Clojure or Java
    + [Scalding](https://github.com/twitter/scalding/wiki/Getting-Started) - An extension to Cascading that enables application development with Scala, a powerful language for solving functional problems
    + [PyCascading](https://github.com/twitter/pycascading) - A Python wrapper for Cascading
- Security
    + [Apache Knox Gateway](https://knox.apache.org/books/knox-0-10-0/dev-guide.html) - A REST API Gateway for interacting with Apache Hadoop clusters
- Operations
    + [Apache Ambari](https://cwiki.apache.org/confluence/display/AMBARI/Ambari) - Tool for provisioning, managing, and monitoring Apache Hadoop clusters
- Massively parallel processing database (MPP)

## Enterprise Big Data and Analytics Products and Services
- [Databricks](https://databricks.com/) - Data integration, real-time exploration, and production pipelines in the cloud, powered by Apache® Spark
- [Talend](http://www.talend.com/) - Open source integration software provider to data-driven enterprises
- [Teradata](http://www.teradata.com/?LangType=1033)
    + Business Analytics Solutions
    + Analytical Architecture Consulting
    + Hybrid Cloud Products
- [Pentaho](http://www.pentaho.com/)
- [Matlab](https://www.mathworks.com/help/matlab/) - The Language of Technical Computing
- [HPE Vertica](http://www8.hp.com/us/en/software-solutions/advanced-sql-big-data-analytics/) - Enables organizations to manage and analyze massive volumes of structured and semi-structured data quickly and reliably with no limits or business compromises
- [IBM SPSS Modeler](https://www.ibm.com/marketplace/cloud/spss-modeler/resources/us/en-us#product-header-top) - A predictive analytics platform that helps you build accurate predictive models quickly and deliver predictive intelligence to individuals, groups, systems and the enterprise
- [IBM SPSS Statistics](https://www.ibm.com/marketplace/cloud/statistical-analysis-and-reporting/us/en-us) - An integrated family of products that addresses the entire analytical process, from planning to data collection to analysis, reporting and deployment
- [SAS](https://support.sas.com/documentation/) - Business intelligence software
- [Alteryx](http://downloads.alteryx.com/documentation.html)
- [Qubole](https://www.qubole.com/)
- [SAP](https://support.sap.com/documentation.html)
- [KNIME](https://tech.knime.org/documentation)
- [Splunk](http://docs.splunk.com/Documentation)
- [FICO Big Data Analyzer - Formerly Karmasphere](http://www.fico.com/en/products/fico-big-data-analyzer#corebenefits)
- [DataScience.com](https://www.datascience.com/) - A powerful platform for enterprise data science
- [DataRobot](https://www.datarobot.com/) - Advanced enterprise machine learning platform

## IoT
- [AWS IoT](https://aws.amazon.com/documentation/iot/) - Easily and securely connect devices to the cloud

## Other
- Load testing
    + [Apache JMeter](http://jmeter.apache.org/) - Java application designed to load test functional behavior and measure performance