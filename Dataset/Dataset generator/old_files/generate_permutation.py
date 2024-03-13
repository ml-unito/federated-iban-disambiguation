import random


def generate_permutations_old(name, rowNumber):
    # Generate aliases by introducing transcription errors.
     # The number of the aliases generated depends by the
     # rowNumber parametes

  words = name.split()
  aliases = []
  newT = T

  # The name is made by more than 1 word
  if len(words) > 1:
    for i in range(rowNumber):
      check = True
      for j in range(len(words)):

        # Add additional spaces
        if check and random.random() < C:
          check = False
          alias_with_spaces = list(words)
          alias_with_spaces.insert(j, '')
          aliases.append(' '.join(alias_with_spaces))
          break

        # Remove a word
        if check and random.random() < C:
          check = False
          alias_without_word = list(words)
          del alias_without_word[j]
          aliases.append(' '.join(alias_without_word))
          break

      if check: aliases.append(name)

  # The name is a single word
  else:
    aliases = [name for i in range(rowNumber)]
    newT = T + (0.3 * T)


  # Introduce transcription errors based on T (Temperature) value
  for j,alias in enumerate(aliases):
    word = list(alias)
    random_positions = random.sample(range(len(word)), random.randint(0, len(word) // 2))
    for i in random_positions:
        if random.random() < newT: word[i] = random.choice([' ', '.', ',', '&', '-', '+'])
    aliases[j] = ''.join(word)

  aliases[0] = name
  return aliases



def generate_permutations_by_name_length(name):
  """ Generate aliases by introducing transcription errors.
      The number of the aliases generated depends by the
      length of name parametes """

  words = name.split()
  aliases = []

  for i in range(len(words)):
    for j in range(i+1, len(words)):

      # Add additional spaces
      if random.random() < C:
        alias_with_spaces = list(words)
        alias_with_spaces.insert(j, '')
        aliases.append(' '.join(alias_with_spaces))
      else: aliases.append(name)

      # Remove a word
      if random.random() < C:
        alias_without_word = list(words)
        del alias_without_word[j]
        aliases.append(' '.join(alias_without_word))
      else: aliases.append(name)


  # Introduce transcription errors based on temperature
  for j,alias in enumerate(aliases):
      word = list(alias)
      random_positions = random.sample(range(len(word)), random.randint(0, len(word)// 2))
      for i in random_positions:
        if random.random() < T: word[i] = random.choice([' ', '.', ',', '&', '-', '+'])
      aliases[j] = ''.join(word)

  aliases[0] = name
  return aliases
