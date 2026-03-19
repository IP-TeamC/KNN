#!/bin/bash

echo -e "\n\n\nJavadoc vorher generieren!!!\n\n"
sleep 5

rm -r target
mvn clean package
cp target/knn-*-jar-with-dependencies.jar knn.jar

zip -r target/KNN_TeamC.zip knn.jar README.md library.md configuration.md *.md pom.xml assets_docs conf data javadoc models src
rm knn.jar