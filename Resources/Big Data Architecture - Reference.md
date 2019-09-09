## Reference Architectures
+ Extended Relational
+ Non-Relational
+ Hybrid

## Architectural Goals, Principles, and Considerations
- Consistency
- [Batch (slow/cold) vs. real-time streaming (fast/hot) data processing and paths](https://www.opsgility.com/blog/2016/11/07/big-data-and-iot-lambda-architecture/)
    + Slow/cold path - batch processing
        * Batch Processing / Analysis
        * Historical Lookup
        * Auditing
    + Fast/hot path - real-time processing
        * Real-time Analytics / Machine Learning Analysis
        * Real-time Reporting
        * Notifications
- Embedded models or interfaces
- API or RPC or REST
- Deployed trained models (offline learning) vs. [online learning](https://en.wikipedia.org/wiki/Online_machine_learning)
+ Latency (near real time)
+ Reliability and fault tolerance
+ Availability
+ Scalability/Volume handling
+ Performance/speed
    * Goals and implementation - Oracle
        - Analyze and transform data in real-time
        - Optimize data structures for intended use
        - Use parallel processing
        - Increase hardware and memory
        - Database configuration and operations
        - Dedicate hardware sandboxes
        - Analyze data at rest, in-place
+ Throughput
+ Extensibility
+ Security
+ Cost/financial
+ Data quality
+ Skills availability
+ Backup and recovery
+ Locations and placement
+ Privacy and sensitive data
+ Disaster recovery
+ Schema on read vs schema on write
    * Bringing the analytical capabilities to the data, VS
    * Bringing the data to the analytical capabilities through staging, extracting, transforming and loading
+ Maturity Considerations - Oracle
    + Reference architecture
    + Development patterns
    + Operational processes
    + Governance structures and polices

## Enterprise Big Data Architectural Components
- Governance
    + Govern data quality
- Operations, Infrastructure, and DevOps
- Monitoring
- Security and privacy
    + Authentication
    + Authorization
    + Accounting
    + Data protection
    + Compliance
- Data Aquisition, Ingestion, and Integration 
    + Messaging and message queues
    + ETL/ELT
    + Change data capture
    + FTP
    * API/ODBC
    * Replication
    * Bulk movement
    * Virtualization
    * Analytics types and options on ingestion - Oracle
        * Sensor-based real-time events
        * Near real-time transaction events
        * Real-time analytics
        * Near real time analytics
        * No immediate analytics
- Data Processing
    + Batch and stream processing/computing (velocity)
        * Massive scaling and processing of multiple concurrent input streams
    + Parallel computing platform
        * Clusters or grids
        * Massively parallel processing (MPP)
        * High performance computing (HPC)
    + Options - Oracle
        * Leave it at the point of capture
        * Add minor transformations
        * ETL data to analytical platform
        * Export data to desktops
    + Fast data - Oracle
        * Streams
        * Events
        * Actions
- Data Access
    + Querying
    + Real-time analytics
    + BI analytics
    + MapReduce analytics
- Data Modeling and Structure
    + Star schema
    + Snowflake schema
- Data Analysis, data mining, discovery, simulation, and optimization
    + Advanced analytics and modeling
    + Text and natural language analytics
    + Video and voice analytics
    + Geospatial analytics
    + Data visualization
    + Data mining
    + Where to do analysis - Oracle
        * At ingest – real time evaluation
        * In a raw data reservoir
        * In a discovery lab
        * In a data warehouse/mart
        * In BI reporting tools
        * In the public cloud
        * On premises
    + Data sets
    + Data science
    + Data discovery
    + In-place analytics
    + Faceted analytics
    + SQL analytics
- Data Storage and Management
    + Data lake
    * Data warehouse (volume), aka enterprise information store
        - Centralized, integrated data store
        - Powers BI analytics, reporting, and drives actionable insights
        - Responsible for integrating data
        - Structured, prepared, and stored data optimized for
            + Analytical applications and decision support
            + Querying and reporting
            + Data mining
        - In-database analytics
        - Operational analytics
        - MPP engine
        - 'Deep analytical appliance' - IBM
    * Operational data store (ODS)
    * Database Systems and DBMS
        - Relational (RDBMS)
        - NoSQL
            + Real-time analytics and insights
        - NewSQL
        - Hybrid
    * Data marts
        - Data warehouse extracted data subsets oriented to speciﬁc business lines, departments or analytical applications
        - Can be a 'live' data mart
    * File systems (Non-distributed)
    * Distributed file systems (e.g., HDFS) and Hadoop (volume and variety)
        - Real-time and MapReduce analytics and insights
        - Deep analysis of petabytes of structured and unstructured data
    * In-memory
    * Data factory
    * Data Reservoir
    * Dedicated and ad-hoc
        - Discovery labs
        - Sandboxes
- Data lifecycle management
    + Rule-based Data and Policy Tracking
    + Data compression
    + Data archiving
- Deployment Choice
    + On-premise, aka traditional IT
    + In-cloud
        * Public cloud
        * Private cloud
    + Appliance
    + Managed services
- Presentation, Analytics, and Applications (visibility)
    + Browser/web
    + Mobile
    + Desktop
    + Dashboards
    + Reports
    + Notifications and messaging
    + Scorecards
    + Charts and graphics
    + Visualization and discovery
    + Search
    + Alerting
    + EPM and BI applications
    + Recommendations

## Enterprise Big Data Components
- http://hortonworks.com/wp-content/uploads/2014/03/11.png

## Big Data Processing Key Functional Capabilities - IBM
- Data ingestion
    + Optimize the process of loading data in the data store to support time-sensitive analytic goals.
- Search and survey
    + Secure federated navigation and discovery across all enterprise content.
- Data transformation
    + Convert data values from source system and format to destination system and format.
- Analytics
    + Discover and communicate meaningful patterns in data.
- Actionable decisions
    + Make repeatable, real-time decisions about organizational policies and business rules.
- Discover and explore
    + Discover, navigate, and visualize vast amounts of structured and unstructured information across many enterprise systems and data repositories.
- Reporting, dashboards, and visualizations
    + Provide reports, analysis, dashboards, and scorecards to help support the way that people think and work.
- Provisioning
    + Deploy and orchestrate on-premises and off-premises components of a big data ecosystem.
- Monitoring and service management
    + Conduct end-to-end monitoring of services in the data center and the underlying infrastructure.
- Security and trust
    + Detect, prevent, and otherwise address system breaches in the big data ecosystem.
- Collaborate and share

## Big data and analytics architecture on cloud - IBM
- Analytics-as-a-service
    + Consumes both data at rest and in motion
    + Applies analytical algorithms
    + Provides
        * Dashboards
        * Reports
        * Visualizations
        * Insights
        * Predictive modeling
    + Abstracts away all complexity of data collection, storage, and cleansing
- Data-as-a-service
    + Data-at-rest-service
    + Data-in-motion-service
- NoSQL tools (Hive, Pig, BigSQL, ...)
- EMR clusters (Hadoop, Cassandra, MongoDB, ...) and Traditional DW
- Big data file system (HDFS, CFS, GPFS, S3, ...)
- Infrastructure & Appliances (Baremetal or IaaS) and object storage

## The Oracle Enterprise Architecture Development Process (OADP)
- Designed to be a flexible and a “just-in-time” architecture development approach
- Key Steps
    + Establish Business Context and Scope
    + Establish an Architecture Vision
    + Assess the Current State
    + Establish Future State and Economic Model
    + Develop a Strategic Roadmap
    + Establish Governance over the Architecture

## Data Storage Functions
- Staging
    + Temporary storage
    + Used for cleaning, integration and transformation routines
- Data management
    + Long-time managed storage
    + Clean and integrated data
- Sandboxing
    + Temporary data stores
    + Used by people, groups, and departments
    + Experimentation with data, processing, and analysis techniques
- Application optimized storage
    + Example usage = data mart
- Archive and raw data archive
    + Raw, processed, and transformed data

## [The 7 V's of Big Data](https://www.impactradius.com/blog/7-vs-big-data/)
+ Volume - Scale of data
+ Variety - Different forms of data
+ Velocity - Analysis of streaming data
+ Veracity - Overall quality and correctness of the data
    * Garbage in, garbage out
    * Assess the truthfulness and accuracy of the data as well as identify missing or incomplete information
- Visibility/Visualization
- Value
- Variability

## Data types and sources
+ Structured
    * Transactions
    * Master and reference
+ Unstructured
    * Text
    * Image
    * Video
    * Audio
    * Social
+ Semi-structured
    * Machine generated
+ Data storage (databases)
+ Sensors
+ Events
+ [Parquet](https://parquet.apache.org/)
+ RFID tags
+ Instore WiFi logs
+ Machine Logs
    * Application
    * Events
    * Server
    * CDRs
    * Clickstream
+ Text, including documents, emails, scanned documents, records, ...
+ Social networks
+ Public web
+ Geo-location/geospatial
+ Feeds
+ Machine generated
+ Clickstream
+ Software
+ Media
    * Images
    * Video
    * Audio
+ Business applications
    * OLTP - Online transaction processing
    * ERP - Enterprise resource planning
    * CRM - Customer relationship management
    * SCM - Supply chain management
    * HR
    * Product/Project management
+ Online chat
+ Merchant listings
+ DMP - Data management platform (advertising/marketing)
+ CDR - Call detail records
+ Surveys, questionnaires, binary questions, and sentiment
+ Billing data
+ Product catalog
+ Network data
+ Subscriber data
+ Staffing
+ Inventory
+ POS and transactional
+ eCommerce transactions
+ Biometrics
+ Mobile devices
+ Weather data
+ Traffic pattern data
+ Mobile devices
+ Surveillance

**Big Data Architecture Patterns**
- [Polyglot](http://datadventures.ghost.io/2014/07/06/polyglot-processing/)
- [Lambda](http://lambda-architecture.net/)
- [Kappa](http://milinda.pathirage.org/kappa-architecture.com/)
- [IOT-A](http://iot-a.info/)
    + Message Queue/Stream Processing (MQ/SP) block
        * Buffer data
            - Processing speed
            - Throughput handling of downstream components
            - Micro-batching can increase ingestion rate into downstream components
        * Process and filter data
            - Cleaning and removal
            - Stream processing
                + Continuous queries
                + Aggregates
                + Counts
                + Real-time machine learning/AI
        * Output
            - Real-time
            - Ingest data into downstream blocks (DB and/or DFS)
        * Example technologies
            - [Kafka](http://kafka.apache.org/)
            - [Spark](http://spark.apache.org/)
            - [Fluentd](http://www.fluentd.org/)
            - [Storm](http://storm.apache.org/)
    + Database (DB) block
        * Provides granular, structured, low-latency access to the data
        * Typically NoSQL
            - MongoDB
            - Cassandra
            - HBase
        * Output
            - Interactive ad-hoc querying
                + Data store API (e.g., HBase, MongoDB, ...)
                + Standard SQL interface
        * Example technologies
            - [Spark](http://spark.apache.org/)
            - [Drill](http://incubator.apache.org/drill/)
    + Distributed File System (DFS) block
        * Batch jobs over entire dataset
            - Aggregations
            - Reporting
            - Integration across data sources
                - E.g., with unstructured data
        * Long term storage (archiving)
        * Example technologies
            - [Hive](http://hive.apache.org/)
            - [Mahout](http://mahout.apache.org/)

## IoT Solution Components
- Connected devices
- Support for a variety of workload types
    + Stream processing
    + Low-latency data queries
    + SLAs (depends on application)
        * [Recovery Point Objective](https://en.wikipedia.org/wiki/Recovery_point_objective)
        * [Recovery Time Objective](https://en.wikipedia.org/wiki/Recovery_time_objective)
        * Availability
        * Latency
        * Disaster recovery
    + Security
    + Privacy
        * ACLs
        * Data encryption and masking
- Data management
- Data modeling and schemas
    + Device metadata
- Data streams
    + Composed of data records flowing through the system
- Processing and output
    + Native and processed raw data support
        * Input data typically time-series
    + Real-time stream
    + Interactive querying
    + Output generated in batches
    + Data transformations
    + Aggregations and computation
    + Data integration and enrichment
- Data movement and storage
    + One or more data stores
- Leverage _big data_ approach
    + Scale out techniques and storage on commodity hardware
        * Historical data/references (_volume_)
    + Schema-on-read (e.g., data lake)
    + Community defined interfaces
    + Many different data formats and non-relational sensor data (_variety_)
    + High rate data generation and handling via data streams in IoT context (_velocity_)
- Analytics
- APIs/SDKs
- Applications and presentation

**Big Data and IoT Tech Stacks**
- SACK
    + Spark - Digest
    + Akka - Ingest
    + Cassandra
    + Kafka
- SMACK
    + Spark
    + Mesos
    + Akka
    + Cassandra
    + Kafka

## Hadoop Benefits
- Built on the _shared nothing_ principle
    + Each node is independent and self-sufficient
- Ability to store any and all data types relatively cheap
- Ability to process any and all data quickly and relatively cheap
- Vast community, ecosystem, and pluggable architecture
- Scalable, flexible, computational model

## Data Processing and access methods/patterns
- Batch
    + Process batches of data on regular time intervals, e.g., hourly, daily, overnight, etc.
    + Aka, MapReduce on Hadoop
- Real-time
    + Monitor and react in real time
    + Key-value data stores, such as NoSQL, allow for high performance, index-based retrieval - Oracle
    + Real-time MapReduce and processing (e.g., Spark)
- [Streaming](http://blog.cloudera.com/blog/2015/06/architectural-patterns-for-near-real-time-data-processing-with-apache-hadoop/)
    + Stream ingestion
    + Near Real-Time (NRT) Event Processing with External Context
    + NRT Event Partitioned Processing
    + Complex Topology for Aggregations or ML
- Interactive/ad-hoc querying
    + Data analysts reviewing data
- Online
- Search
- In-memory

## Offline vs Online Learning

Coming soon...

## General References
- [Lambda Architecture](http://lambda-architecture.net/)
- [AWS Architecture Center](https://aws.amazon.com/architecture/?nc1=f_cc)
- [AWS Big Data Partner Solutions](https://aws.amazon.com/big-data/partner-solutions/)
- [GCP Architecture](https://cloud.google.com/docs/tutorials#architecture)
- [Introduction to big data classification and architecture](http://www.ibm.com/developerworks/library/bd-archpatterns1/)
- [An Enterprise Architect’s Guide to Big Data](http://www.oracle.com/technetwork/topics/entarch/articles/oea-big-data-guide-1522052.pdf)
- [BIG DATA REFERENCE ARCHITECTURE](https://thinkbiganalytics.com/leading_big_data_technologies/big-data-reference-architecture/)
- [Getting Started with Big Data Architecture](http://blog.cloudera.com/blog/2014/09/getting-started-with-big-data-architecture/)
- [BIG DATA: Architectures and Technologies](https://www.sei.cmu.edu/go/big-data/)
- [Big Data Architecture](http://bigdata.teradata.com/US/Big-Ideas/Big-Data-Architecture/)
- [Big Data Analytics Architecture](http://www.thebigdatainsightgroup.com/site/sites/default/files/Teradata's%20-%20Big%20Data%20Architecture%20-%20Putting%20all%20your%20eggs%20in%20one%20basket.pdf)
- [What is Streaming Data?](https://aws.amazon.com/streaming-data/)

## Big Data Best Practices - Oracle
- Align Big Data with Specific Business Goals
- Ease Skills Shortage with Standards and Governance
- Optimize Knowledge Transfer with a Center of Excellence
- Top Payoff is Aligning Unstructured with Structured Data
- Plan Your Discovery Lab for Performance
- Align with the Cloud Operating Model

## Architecture Principles - Oracle
- Accommodate All Forms of Data
- Consistent Information and Object Model
- Integrated Analysis
- Insight to Action

## IBM Data Governance Council Maturity Model
- Organizational Structures & Awareness
- Stewardship
- Policy
- Value Creation
- Data Risk Management & Compliance
- Information Security & Privacy
- Data Architecture
- Data Quality Management
- Classification & Metadata
- Information Lifecycle Management
- Audit Information, Logging & Reporting

## Diagrams
<figure>
    <img src='http://www.datazoomers.com/wp-content/uploads/2013/02/bi_data_ware_house.jpg' alt='missing' />
    <figcaption>Courtesty of DataZoomers</figcaption>
</figure>
<br/>
<figure>
    <img src='http://cdn.guru99.com/images/ETL_Testing/ETLTesting_1.jpg' alt='missing' />
    <figcaption>Courtesty of Guru99</figcaption>
</figure>

<!-- <figure>
    <img src='http://cdn.guru99.com/images/ETL_Testing/ETLTesting_4.jpg' />
    <figcaption>Courtesty of Guru99</figcaption>
</figure> -->
