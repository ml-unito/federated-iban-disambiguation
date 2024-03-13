import random
import editdistance


abbreviation_variations = [
    
    #"Società Internazionali
    ["Corp.", "Corp", "Corpn.", "Cp."], #"Corporation": 
    ["Inc.", "Inc", "Incorporated"], #"Incorporated": 
    ["Ltd.", "Ltd", "Limited", "Ltée", "LTD", "LTd.", "L.t.d.", "LTT",],  #"Limited": 
    ["Co.", "Co", "Company", "Comp."], #"Company": 
    ["P.L.C.", "Public Limited Company", "PLC ", "plc", "pLC", "PLCC", "P.L.C", "Pvt. Ltd.", "Pvt Ltd", "Pvt. Limited", "P.L.C.", "Public Ltd.", "Pvt. Ltd"],  #"Private Limited": 
    
    #"Società Italiane
    ["S.p.A.", "SpA", "S.P.A.", "SPA", "Sp.A.", "S.P.A", "Società per Azioni", "SpA ",],
    ["S.r.l.", "S.R.L.", "SRL", "S.r.L", "S r l", "Srl"],
    ["S.a.p.a.", "SAPA", "S.A.P.A.", "S.a.p.a", "S.A.P.A"],
    ["S.a.s.", "SAS", "S.A.S.", "S.a.s", "S.A.S"],
    ["S.c.", "S.C.", "SC", "S.c", "S.C"],
    ["S.a.", "SA", "S.A.", "S.a", "S.A"],
    ["S.c.a.r.l.", "S.C.A.R.L.", "SCARL", "S.c.a.r.l", "S.C.A.R.L"],
    ["S.d.f.", "S.D.F.", "SDF", "S.d.f", "S.D.F"],
    ["S.u.", "S.U.", "SU", "S.u", "S.U"],
    ["S.N.C.", "SNC", "S.n.c", "S.N.C"],
]

Transcription_errors = ['.', ',', '-', '/', '- ', ". ", " .", " - "]
international_company_words = [
    "Group",
    "Corporation",
    "Holdings",
    "Enterprises",
    "International",
    "Global",
    "Solutions",
    "Services",
    "Technologies",
    "Industries",
    "Partners",
    "Systems",
    "Worldwide",
    "Ventures",
    "Enterprises",
    "Brothers",
    "Sons",
    "Company",
    "Associates",
    "Limited"
]


def compute_variation_abbreviation(words, threshold):
  """ """
  newList = []
  for j in range(len(words)):
    selectedList = []
    for abbreviations in abbreviation_variations:
      abbrLower = [el.lower() for el in abbreviations]
      
      for ab in abbrLower:
        if editdistance.eval(words[j].lower(), ab) < threshold:
          index = abbrLower.index(ab)
          if selectedList == []: selectedList = [el for (i,el) in enumerate(abbreviations) if i != index]
          else: selectedList += [el for (i,el) in enumerate(abbreviations[:len(abbreviations) // 2]) if i != index]
                    
    if len(selectedList) > 0: newList.append((words[j], selectedList))

  return newList



def compute_transcription_errors(aliasList, newT):
  """ Introduce transcription errors based on T (Temperature) value """

  aliases = aliasList
  for j,alias in enumerate(aliases):
    
    word = list(alias)
    if len(alias.split()) > 1:
      voidPositions = [j for j,i in enumerate(word) if i == ' ']
      for i in voidPositions:
          if random.random() < newT: word[i] = random.choice(Transcription_errors)
    else:
      random_positions = random.sample(range(len(word)), random.randint(0, len(word) // 2))
      for i in random_positions:
          if random.random() < newT: word[i] = random.choice([el for el in Transcription_errors if el != "/"])

    aliases[j] = ''.join(word)

  return aliases


def introduce_variability(aliasList, abbrList, V):
  """  Introduce variability in companies abbreviations """

  aliases = aliasList
  for j,_ in enumerate(aliases):
    if len(abbrList) > 0:
      elem = random.choice(abbrList)
      elemList = random.choice(elem[1])
      if random.random() < V:
        aliases[j] = aliases[j].replace(elem[0], elemList)
      else:
          newAlias = aliases[j].split(elem[0])
          if newAlias[0] != "": aliases[j] = newAlias[0].strip()
          elif newAlias[1] != "": aliases[j] = newAlias[1].strip()
          else: continue
          
  return aliases


def introduce_white_spaces(aliasList, C):
  """ """

  aliases = aliasList
  # Add additional spaces
  for j,alias in enumerate(aliases):
    if random.random() < C:
      alias_with_spaces = list(alias)
      voidPositions = [k for k,q in enumerate(alias_with_spaces) if q == ' ']
      if len(voidPositions) > 0:
        el = random.choice(voidPositions)
        alias_with_spaces.insert(el, ' ')
        aliases[j] = ''.join(alias_with_spaces)
  
  return aliases

def generate_permutations(name, rowNumber, T, C, V, Edit_threshold):
  """ Generate aliases by introducing transcription errors.
      The number of the aliases generated depends by the
      rowNumber parametes """

  words = name.split()
  if "-" in name:
    name = name.replace("-", "")
    words = name.split("-")
  aliases = []
  newT = T

  # The name is made by more than 1 word
  if len(words) > 2:
    for i in range(rowNumber):
      check = True
      for j in range(len(words)):
        
        if j in range(2, len(words)):
          # Remove a word
          if random.random() < C:
            check = False
            alias_without_word = list(words)
            del alias_without_word[j]
            aliases.append(' '.join(alias_without_word))
            break

      if check: aliases.append(name)

  # The name is a single word
  else:
    if random.random() < V:
      ab = random.choice(abbreviation_variations)
      name += " " + random.choice(ab)
      words = name.split()
    aliases = [name for _ in range(rowNumber)]
    newT = T + (T * 0.8)

  abbrList = compute_variation_abbreviation(words, Edit_threshold)
  aliases = compute_transcription_errors(aliases, newT)
  aliases = introduce_variability(aliases, abbrList, V)
  aliases = introduce_white_spaces(aliases, C)

  aliases[0] = name
  return aliases