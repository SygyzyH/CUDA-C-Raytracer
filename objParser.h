#include <iostream>
#include <fstream>
#include <cstdarg>
#include <string>
#include <vector>
#include <tgmath.h>
#include "prim/types.h"

#ifndef OBJPARSER_H
#define OBJPARSER_H

std::vector<poly *> parseObjectFiles(int fileCount, ...) {
    va_list fileNames;
    va_start(fileNames, fileCount);

    std::vector<poly *> facesVec;

    for (int currentFileIndex = 0; currentFileIndex < fileCount; currentFileIndex++) {
        std::vector<vec *> verteciesVec;

        // open file
        std::string txt;
        std::fstream currentFile(va_arg(fileNames, const char*));

        // parse file
        while (getline(currentFile, txt)) {
            if (txt.find("/") != std::string::npos) {
                std::cout << "Unable to parse file: " << "User-defined normals and texture mapping are currently not supported.\n";
                return facesVec;
            }

            switch (txt[0]) {
                case 'v': {
                    vec *newVertex = (vec *) malloc(sizeof(vec));
                    newVertex->value = (float *) malloc(3 * sizeof(float));
                    newVertex->size = 3;

                    int lastSpaceChar = 1;
                    for (int i = 0; i < 3; i++) {
                        std::string::size_type nextSpaceChar = txt.find(' ', lastSpaceChar + 1);
                        if (!nextSpaceChar)
                            nextSpaceChar = txt.length() - 1;
                        newVertex->value[i] = std::stof(txt.substr(lastSpaceChar + 1, nextSpaceChar - 1));
                        lastSpaceChar = nextSpaceChar;
                    }

                    verteciesVec.push_back(newVertex);
                    break;
                }
                case 'f': {
                    int vertexIndecies = 1000;

                    int lastSpaceChar = 1;
                    for (int i = 0; i < 3; i++) {
                        std::string::size_type nextSpaceChar = txt.find(' ', lastSpaceChar + 1);
                        if (!nextSpaceChar)
                            nextSpaceChar = txt.length() - 1;
                        vertexIndecies += pow(10, i) * (std::stoi(txt.substr(lastSpaceChar + 1, nextSpaceChar - 1)) - 1);
                        lastSpaceChar = nextSpaceChar;
                    }

                    poly *newPoly = (poly *) malloc(sizeof(poly));
                    newPoly->vert1 = verteciesVec[vertexIndecies % 10];
                    newPoly->vert2 = verteciesVec[vertexIndecies / 10 % 10];
                    newPoly->vert3 = verteciesVec[vertexIndecies / 100 % 10];
                    newPoly->color = 0x007f7f7f;
                    newPoly->reflectivity = 0.5f;

                    facesVec.push_back(newPoly);
                    break;
                } 
            }
        }

        // close file
        currentFile.close();
    }

    va_end(fileNames);

    return facesVec;
}

#endif