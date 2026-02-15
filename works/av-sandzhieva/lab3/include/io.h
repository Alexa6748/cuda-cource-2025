#ifndef IO_H
#define IO_H

unsigned char* readPGM(const char* filename, int* width, int* height);
void writePGM(const char* filename, unsigned char* data, int width, int height);

unsigned char* readBMP(const char* filename, int* width, int* height);
void writeBMP(const char* filename, unsigned char* data, int width, int height);

#endif // IO_H
