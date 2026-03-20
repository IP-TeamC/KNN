#!/bin/bash

echo -e "\n\n\nJavadoc vorher generieren!!!\n\n"
sleep 5

rm -r target
mvn clean package
cp target/knn-*-jar-with-dependencies.jar knn.jar

zip -r target/KNN_TeamC.zip knn.jar README.md pom.xml docs conf data models src .run
rm knn.jar