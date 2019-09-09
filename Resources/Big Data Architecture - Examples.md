## Architecture Guides, Usage, and Examples

**AWS**
- [Solution Development Guides](https://aws.amazon.com/solutions)
    + Reference Architectures
        * [Web Application Hosting](http://media.amazonwebservices.com/architecturecenter/AWS_ac_ra_web_01.pdf)
        * [Batch Processing](http://media.amazonwebservices.com/architecturecenter/AWS_ac_ra_batch_03.pdf)
        * [Large Scale Computing and Large Data Sets](http://media.amazonwebservices.com/architecturecenter/AWS_ac_ra_largescale_05.pdf)
        * [Time Series Processing](http://media.amazonwebservices.com/architecturecenter/AWS_ac_ra_timeseriesprocessing_16.pdf)
    + By Application
        * [Websites](https://aws.amazon.com/websites/)
        * [Backup and Recovery](https://aws.amazon.com/backup-recovery/)
        * [Archiving](https://aws.amazon.com/archive/)
        * [DevOps](https://aws.amazon.com/devops/)
        * [Big Data](https://aws.amazon.com/big-data/)
        * [High Performance Computing](https://aws.amazon.com/hpc/)
        * [Internet of Things](https://aws.amazon.com/iot/)
        * [Business Applications](https://aws.amazon.com/business-applications/)
        * [Content Delivery](https://aws.amazon.com/content-delivery/)
        * [Mobile Services](https://aws.amazon.com/mobile/)
        * [Scientific Computing](https://aws.amazon.com/government-education/scientific-computing1/)
        * [E-commerce](https://aws.amazon.com/ecommerce-applications/)
    + By Industry Sector
        * [Financial Services](https://aws.amazon.com/financial-services/)
        * [Digital Marketing](https://aws.amazon.com/digital-marketing/)
        * [Media and Entertainment](https://aws.amazon.com/digital-media/)
        * [Gaming](https://aws.amazon.com/game-hosting/)
        * [Enterprise IT](https://aws.amazon.com/enterprise/)
        * [Healthcare & Life Sciences](https://aws.amazon.com/health/)
        * [Government](https://aws.amazon.com/government-education/government/)
        * [Nonprofit](https://aws.amazon.com/government-education/nonprofits/)
        * [Education](https://aws.amazon.com/education/)
- [Partner Solutions](https://aws.amazon.com/solutions/partners/)
    + [Big Data](https://aws.amazon.com/big-data/partner-solutions/)
    + [Storage](https://aws.amazon.com/backup-recovery/partner-solutions/)
    + [DevOps](https://aws.amazon.com/devops/partner-solutions/)
- [Case Studies](https://aws.amazon.com/solutions/case-studies/)
    + [Analytics](https://aws.amazon.com/solutions/case-studies/analytics/)
    + [Big Data](https://aws.amazon.com/solutions/case-studies/big-data/)
    + [Enterprise](https://aws.amazon.com/solutions/case-studies/enterprise-it/)
    + [Startups](https://aws.amazon.com/solutions/case-studies/start-ups/)
    + [Web Apps](https://aws.amazon.com/solutions/case-studies/web-mobile-social/)

**Google Cloud Platform**
- [Big data reference architecture diagram](https://cloud.google.com/images/products/big-data/big-data-diagram.png)
- [Solution Development Guides](https://cloud.google.com/solutions/)
    + [Media](https://cloud.google.com/solutions/media/)
    + [Mobile Applications](https://cloud.google.com/solutions/mobile/#development_guides)
    + [Big Data](https://cloud.google.com/solutions/big-data/#development_guides)
    + [Financial Services](https://cloud.google.com/solutions/financial-services/#development_guides)
    + [Gaming](https://cloud.google.com/solutions/gaming/#development_guides)
    + [Retail & Commerce](https://cloud.google.com/solutions/commerce/#development_guides)
    + [Internet of Things](https://cloud.google.com/solutions/iot/#development_guides)
    + [Websites and Web Apps](https://cloud.google.com/solutions/websites/#development_guides)
    + [Development & Test](https://cloud.google.com/solutions/dev-test/#development_guides)

**By Technology**
- [Apache Storm](http://storm.apache.org/releases/current/Powered-By.html)
- [Apache Spark](http://www.datanami.com/2014/03/06/apache_spark_3_real-world_use_cases/)
- [Apache Mahout](https://mahout.apache.org/general/powered-by-mahout.html)

**IoT**

Coming soon...

## Oracle Architecture and Patterns Examples

## IBM Architecture and Patterns Examples

**Solution Patterns - IBM**
- Landing Zone Warehouse
- Virtual Tables
- Discovery Tables
- Streams Dynamic Warehouse
- Streams Detail with Update
- Direct Augmentation
- Warehouse Augmentation
- Streams Augmentation
- Dynamic Search Cube

**Component Patterns - IBM**
- Source Data
- Source Event
- Landing Area Zone ETL
    + Extract
    + Normalize
    + Clean
- Landing Area Zone Search and Survey
    + Find
    + Filter
    + Extract
- Landing Area Zone Stream Filter
- Landing Area Zone Stream Augmentation
- Landing Area Zone Warehouse Augmentation
- Landing Area Zone Index
- Exploration Mart
- Analytics Mart
- Report Mart
- Virtual Report Mart
- Virtual Search Mart
- Predictive Analytics

**Big Data Exploration Example Architecture - IBM**
- Applications layer
    + Consists of
        * Visualization
        * Discovery
        * Analysis
        * Reporting
        * Statistics
        * Text and entity analytics
    + Access
        * SQL
        * MDX
        * Search
        * REST
- Discovery and assembly layer
    + Consists of
        * Virtual search mart
            - Faceted search
        * Analytics mart
            - Report mart
            - Discovery table
            - Search and survey
        * Report mart
            - ETL
            - Analytics
            - Streams
    + Access
        * NoSQL
        * SQL
        * Search
        * REST
- Landing layer
    + Consists of
        * Shared warehouse and ETL
            - Extract
            - Provision
    + Access
        * Search
        * REST
        * SQL
        * Files
- Source layer
    + Sensors and telemetry
    + Internet
    + Social media
    + Public data
    + Enterprise data
    + ...
