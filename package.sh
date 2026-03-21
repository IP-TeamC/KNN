#!/bin/bash

echo -e "\n\n\nJavadoc (docs/javadoc) und Test-Coverage (docs/test_coverage) vorher generieren!!!\n\n"
sleep 5

rm -r target
mvn clean package
cp target/knn-*-jar-with-dependencies.jar knn.jar

zip -r target/KNN_TeamC.zip knn.jar README.md pom.xml .run conf data docs models src
rm knn.jar
rm -r docs/javadoc docs/test_coverage