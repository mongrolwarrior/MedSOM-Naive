#pragma once
#include <random>
#include "compat.h"

// DATA PARAMETERS
unsigned long article_filter = 1;
// DISPLAY PARAMETERS
ulong showMatrixLimit = 10;
// NETWORK PARAMETERS
ulong dimX = 350;
ulong dimY = 350;
ulong numberOfMeshCategories;
ulong numberOfArticles;
// TRAINING PARAMETERS
ulong radius_end = 1;
float std_coeff = 0.5;
ulong number_of_epochs = 20;
ulong number_of_descending_radii = 12;
bool runFullProgram = true;
bool mapArticleMatches = false;
bool save_selected_article_mesh = false;
bool save_codebook_top10s = false;

// CALCULATED PARAMETERS
ulong radius_start = std::min(dimX, dimY) / 2;
unsigned long long codebookSize;

// GPU PARAMETERS
int blocks = 1024;
int threads = 512;

std::default_random_engine geni(12345678);
std::default_random_engine genu(87654321);
