// read unistd.h file as text and parse it to get system call numbers and names
// then print them to the screen

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 256
#define MAX_SYSCALLS 512

int main(){
    FILE *fp;
    char line[MAX_LINE_LENGTH];
    char *token;
    char syscall_names[MAX_SYSCALLS][MAX_LINE_LENGTH];
    int syscall_numbers[MAX_SYSCALLS];

    fp = fopen("unistd.h", "r");
    if(fp == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    int i = 0;
    int end = 0;
    while(fgets(line, MAX_LINE_LENGTH, fp) != NULL){
        token = strtok(line, " ");
        while(token != NULL){
            if(strcmp(token, "#define") == 0){
                token = strtok(NULL, " ");
                if(strncmp(token, "__NR_", 5) == 0){
                    strcpy(syscall_names[i], token);
                    if(strcmp(token, "__NR_syscalls") == 0) end = 1;
                    token = strtok(NULL, " ");
                    syscall_numbers[i] = atoi(token);
                    i++;
                }
            }
            token = strtok(NULL, " ");
        }

        if (end == 1) break;
    }

    // save syscall numbers to a file
    FILE *fp2;
    fp2 = fopen("syscall_numbers.txt", "w");
    if(fp2 == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    fprintf(fp2, "{");
    for(int j = 0; j < i; j++){
        if (j > 0){
            fprintf(fp2, ", ");
        }
        fprintf(fp2, "%d", syscall_numbers[j]);
    }
    fprintf(fp2, "}");

    fclose(fp2);

    // save syscall names to a file
    FILE *fp3;
    fp3 = fopen("syscall_names.txt", "w");
    if(fp3 == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    fprintf(fp3, "{");
    for(int j = 0; j < i; j++){
        if (j > 0){
            fprintf(fp3, ", ");
        }
        fprintf(fp3, "\"%s\"", syscall_names[j]);
    }
    fprintf(fp3, "}");

    fclose(fp3);

    return 0;
}