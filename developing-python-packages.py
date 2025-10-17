# Write the amended function here
# Open the text file
def count_words(filepath, words_list):
    with open(filepath) as file:
        text = file.read()

    n = 0
    for word in text.split():
        # Count the number of times the words in the list appear
        if word.lower() in words_list:
            n += 1

    print('Lewis Carroll uses the word "cat" {} times'.format(n))
    return n