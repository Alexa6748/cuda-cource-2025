#include "../include/io.h"
#include "../include/utils.h"

unsigned char* readPGM(const char* filename, int* width, int* height)
{
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return nullptr;
    }

    char magic[3] = {0};
    if (fscanf(file, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Not a binary PGM file\n");
        fclose(file);
        return nullptr;
    }

    int ch;
    while ((ch = fgetc(file)) == '#') {
        while (fgetc(file) != '\n' && !feof(file));
    }
    if (!feof(file)) ungetc(ch, file);

    int max_val;
    if (fscanf(file, "%d %d %d", width, height, &max_val) != 3) {
        fprintf(stderr, "Invalid PGM header\n");
        fclose(file);
        return nullptr;
    }

    fgetc(file);

    size_t num_pixels = static_cast<size_t>(*width) * (*height);
    unsigned char* img_data = static_cast<unsigned char*>(malloc(num_pixels));
    if (!img_data || fread(img_data, 1, num_pixels, file) != num_pixels) {
        fprintf(stderr, "Failed to read pixel data\n");
        free(img_data);
        fclose(file);
        return nullptr;
    }

    fclose(file);
    return img_data;
}

void writePGM(const char* filename, unsigned char* data, int width, int height)
{
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Cannot create file %s\n", filename);
        return;
    }

    fprintf(file, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, file);
    fclose(file);
}

unsigned char* readBMP(const char* filename, int* width, int* height)
{
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open BMP file %s\n", filename);
        return nullptr;
    }

    unsigned char header[54];
    if (fread(header, 1, 54, file) != 54 || header[0] != 'B' || header[1] != 'M') {
        fprintf(stderr, "Invalid BMP header\n");
        fclose(file);
        return nullptr;
    }

    int data_offset;
    memcpy(&data_offset, &header[10], sizeof(int));

    memcpy(width, &header[18], sizeof(int));
    memcpy(height, &header[22], sizeof(int));

    
    short bpp;
    memcpy(&bpp, &header[28], sizeof(short));
    if (bpp != 24) {
        fprintf(stderr, "Only 24-bit BMP supported\n");
        fclose(file);
        return nullptr;
    }

    int row_padded = ((*width * 3) + 3) & ~3;
    unsigned char* rgb = static_cast<unsigned char*>(malloc(row_padded * *height));
    if (!rgb) {
        fclose(file);
        return nullptr;
    }

    fseek(file, *(int*)&header[10], SEEK_SET);
    if (fread(rgb, 1, row_padded * *height, file) != static_cast<size_t>(row_padded * *height)) {
        fprintf(stderr, "Failed to read BMP pixels\n");
        free(rgb);
        fclose(file);
        return nullptr;
    }
    fclose(file);

    unsigned char* gray = static_cast<unsigned char*>(malloc(*width * *height));
    if (!gray) {
        free(rgb);
        return nullptr;
    }

    for (int y = 0; y < *height; ++y) {
        for (int x = 0; x < *width; ++x) {
            int idx = y * row_padded + x * 3;
            unsigned char b = rgb[idx];
            unsigned char g = rgb[idx + 1];
            unsigned char r = rgb[idx + 2];
            unsigned char val = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
            gray[(*height - 1 - y) * *width + x] = val;
        }
    }

    free(rgb);
    return gray;
}

void writeBMP(const char* filename, unsigned char* data, int width, int height)
{
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Cannot create BMP file %s\n", filename);
        return;
    }

    int row_padded = ((width * 3) + 3) & ~3;
    int file_size = 54 + row_padded * height;

    unsigned char header[54] = {0};
    header[0] = 'B'; header[1] = 'M';

    int offset = 54;
    int info_size = 40;
    short planes = 1;
    short bits = 24;

    memcpy(&header[2], &file_size, sizeof(int));
    memcpy(&header[10], &offset, sizeof(int));
    memcpy(&header[14], &info_size, sizeof(int));
    memcpy(&header[18], &width, sizeof(int));
    memcpy(&header[22], &height, sizeof(int));
    memcpy(&header[26], &planes, sizeof(short));
    memcpy(&header[28], &bits, sizeof(short));

    fwrite(header, 1, 54, file);

    unsigned char* row_buffer = static_cast<unsigned char*>(malloc(row_padded));
    memset(row_buffer, 0, row_padded);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char gray = data[(height - 1 - y) * width + x];
            row_buffer[x * 3] = gray;
            row_buffer[x * 3 + 1] = gray;
            row_buffer[x * 3 + 2] = gray;
        }
        fwrite(row_buffer, 1, row_padded, file);
    }

    free(row_buffer);
    fclose(file);
}
