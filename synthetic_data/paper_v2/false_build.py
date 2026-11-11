import json
from pathlib import Path
D=Path(__file__).parent
seed=json.loads((D/'seeds.json').read_text())['trees'][:15]
roots=[('TREE PARTS','BARK ROOT TRUNK BRANCH LEAF LIMB BOUGH CROWN'),('GARMENTS','DRESS COAT SUIT CAP TOP TIE VEST SLIP'),('GEOGRAPHIC FEATURES','BANK CAPE SOUND CHANNEL DELTA PLAIN GULF POINT'),('FIRE VOCABULARY','SPARK FLAME MATCH ASH COAL SMOKE KINDLE LIGHT'),('MUSIC TERMS','PITCH NOTE KEY SCALE REST STAFF BAR BEAT'),('CAR PARTS','BOOT HOOD TRUNK BUMPER WHEEL TIRE DASH MIRROR'),('CARD TERMS','ACE KING QUEEN JACK HEART DIAMOND CLUB SPADE'),('JEWELRY TERMS','BAND STUD HOOP CHAIN LINK PEARL ROCK CHARM'),('TELEVISION TITLES','GLEE FRIENDS LOST HOUSE BONES FARGO CHEERS SUCCESSION'),('COLOR NAMES','GOLD SILVER ROSE CORAL SALMON TAN CREAM PLUM'),('WILD CATS','TIGER LION LEOPARD JAGUAR PUMA PANTHER LYNX COUGAR'),('BOARD GAMES','CHESS GO RISK TROUBLE CLUE LIFE SORRY OPERATION'),('BIRDS','CRANE SWIFT SWALLOW DUCK HAWK RAVEN ROOK KITE'),('MAGIC VOCABULARY','CAST SPELL TRICK CHARM WAND HEX JINX CURSE'),('FOOTWEAR','PUMP FLAT WEDGE MULE CLOG SLIDE BOOT LOAFER')]
(D/'false_root_pools.json').write_text(json.dumps([{'id':s['id'],'words':w.split()} for s,(c,w) in zip(seed,roots)],indent=2)+'\n')
# Alternate-sense pools authored only after observing the actual root selections.
raw=[
[('LUGGAGE','TRUNK CASE SATCHEL VALISE SUITCASE DUFFEL HOLDALL BACKPACK','Travel containers'),('DIVERGE','BRANCH FORK SPLIT DEVIATE VEER SEPARATE BIFURCATE PART','Separate from a common course'),('___ THROUGH','LEAF FLIP THUMB SKIM SCAN BROWSE RIFFLE PAGE','Verbs preceding THROUGH when looking through reading material'),('BODY PARTS','LIMB ARM LEG HAND FOOT HEAD NECK TORSO','Anatomical regions or appendages')],
[('FOOD PREPARATION VERBS','DRESS BASTE SEASON MARINATE BRINE GLAZE RUB STUFF','Actions used to prepare or flavor food'),('COVER','COAT COVER SHEATHE ENCASE OVERLAY WRAP ENVELOP BLANKET','Put a covering over something'),('LEGAL PROCEEDINGS','SUIT ACTION CASE APPEAL TRIAL HEARING INQUEST ARBITRATION','Named forms of legal proceeding'),('EQUITY AWARD VERBS','VEST GRANT EXERCISE FORFEIT SELL HOLD TRANSFER CONVERT','Actions or events involving employee share awards')],
[('SHOULDER COVERINGS','CAPE CLOAK SHAWL STOLE PONCHO WRAP MANTLE TIPPET','Garments worn over the shoulders'),('GREEK LETTERS','DELTA ALPHA BETA GAMMA EPSILON ZETA ETA THETA','Names of Greek alphabet letters'),('UNADORNED','PLAIN SIMPLE BARE AUSTERE SPARE UNDECORATED UNEMBELLISHED SEVERE','Adjectives describing an unornamented appearance'),('SEPARATION','GULF GAP CHASM DIVIDE RIFT BREACH SCHISM SPLIT','Words for figurative divisions between people or positions')],
[('CRITICIZE HARSHLY','FLAME BLAST SLAM PAN ROAST RIP BASH TRASH','Verbs meaning attack verbally'),('TREES','ASH OAK ELM BIRCH PINE FIR CEDAR MAPLE','Names of trees'),('___ BLACK','COAL JET INK PITCH NIGHT RAVEN EBONY SOOT','Words used before BLACK to describe a very dark shade'),('___ SIGNAL','SMOKE HAND TRAFFIC TURN RADIO DISTRESS WARNING START','Words preceding SIGNAL in established phrases')],
[('THROW','PITCH HURL TOSS FLING CAST CHUCK LOB SLING','Verbs for propelling something through the air'),('___BOOK','NOTE COOK TEXT WORK GUEST PHONE SCRAP PASS','Words combining with BOOK into a familiar book type'),('___HOLE','KEY BUTTON PEEP PORT LOOP SINK ARM POT','Words combining with HOLE to name an opening or cavity'),('EXHAUST','BEAT DRAIN TIRE EXHAUST FATIGUE WEARY SAP DEBILITATE','Verbs meaning wear out or weaken')],
[('ANIMAL APPENDAGES','TRUNK TAIL HORN WING FIN FLIPPER TENTACLE ANTENNA','External projecting body parts of animals'),('ABUNDANT','BUMPER RECORD PLENTIFUL COPIOUS BOUNTIFUL LAVISH AMPLE ABUNDANT','Adjectives describing a large yield or supply'),('___HOUSE','WHEEL GREEN LIGHT CLUB SCHOOL COURT POWER WARE','Words that form a familiar compound ending HOUSE'),('WEAR OUT','TIRE EXHAUST FATIGUE DRAIN WEARY SAP TAX ENERVATE','Verbs meaning deplete strength')],
[('SUCCEED','ACE SHINE EXCEL TRIUMPH PREVAIL FLOURISH THRIVE SUCCEED','Verbs for doing very well or succeeding'),('NOVELISTS','KING AUSTEN ORWELL WOOLF DICKENS HARDY ELIOT SWIFT','Surnames of established novelists'),('BANDS','QUEEN KISS RUSH BLUR CREAM OASIS JOURNEY CHIC','Single-word names of musical bands'),('GARDEN TOOLS','SPADE RAKE HOE TROWEL SHOVEL SICKLE DIBBER FORK','Hand tools used in a garden')],
[('COLLECTIVE GROUPS','BAND FLOCK HERD SWARM PACK TROOP BEVY GAGGLE','Collective nouns for assembled beings'),('___ JAM','PEARL TRAFFIC PAPER LOG TOE DOOR RASPBERRY STRAWBERRY','Words preceding JAM in established expressions'),('___ STAR','ROCK POP MOVIE CHILD NORTH LODE FALLING SHOOTING','Words preceding STAR in familiar expressions'),('FASCINATE','CHARM ENCHANT CAPTIVATE BEGUILE ENTRANCE FASCINATE SPELLBIND MESMERIZE','Verbs for holding someone spellbound')],
[('JOY','GLEE JOY DELIGHT ELATION BLISS MIRTH JUBILATION EXULTATION','Nouns for intense pleasure or happiness'),('COMPANIONS','FRIENDS ALLIES PALS BUDDIES CHUMS COMPANIONS CRONIES CONFIDANTS','People with personal or supportive relationships'),('UNLOCATED','LOST MISSING ABSENT GONE MISPLACED UNTRACEABLE VANISHED STRAY','Adjectives describing someone or something not where expected'),('AMERICAN CITIES','FARGO PHOENIX MOBILE BUFFALO AUSTIN SALEM DOVER LINCOLN','Names of United States cities')],
[('___ RUSH','GOLD SUGAR HEAD LAND ADRENALINE HOUR MORNING EVENING','Words preceding RUSH in established expressions'),('___ LINING','SILVER FLEECE FUR SILK COTTON SATIN WOOL THERMAL','Words preceding LINING in established expressions'),('ANAGRAMMED ANIMALS','TAN ACT GOD TAR LOW BARE FLOW SHORE','Rearrange the letters of each given word to obtain an animal name.'),('BEAT DECISIVELY','CREAM ROUT TROUNCE DRUB WALLOP CLOBBER THRASH SMASH','Informal verbs for defeating an opponent badly')],
[('___ SHARK','TIGER BULL NURSE WHALE LEMON REEF BASKING HAMMERHEAD','Words preceding SHARK to name shark types'),('___HEART','LION SWEET BRAVE LIGHT HEAVY BROKEN BLEEDING PURPLE','Words that precede HEART in familiar compounds or expressions'),('___ PRINT','LEOPARD ANIMAL FINGER FOOT BLUE NEWS SCREEN SMALL','Words preceding PRINT in familiar open or closed compounds'),('CAR BRANDS','JAGUAR FORD AUDI HONDA BUICK DODGE MINI SMART','Automobile marques')],
[('TAKE A CHANCE','RISK GAMBLE VENTURE WAGER BET CHANCE HAZARD SPECULATE','Verbs involving uncertain outcomes'),('BOTHER','TROUBLE ANNOY IRK VEX PESTER HARASS BUG BOTHER','Verbs meaning disturb someone'),('DETECTIVE EVIDENCE','CLUE FINGERPRINT FOOTPRINT FIBER HAIR TRACE SAMPLE RECORDING','Items that may provide evidence in an investigation'),('___BOAT','LIFE ROW SAIL SPEED HOUSE STEAM MOTOR TUG','Words combining with BOAT to name a vessel type')],
[('CONSTRUCTION MACHINES','CRANE EXCAVATOR BULLDOZER GRADER BACKHOE LOADER COMPACTOR SCRAPER','Heavy equipment used at building sites'),('AVOID','DUCK EVADE DODGE ELUDE SHUN ESCHEW SIDESTEP CIRCUMVENT','Verbs meaning keep away from something'),('SELL','HAWK PEDDLE VEND MARKET RETAIL PURVEY TRADE SELL','Verbs meaning offer goods for purchase'),('QUADRILATERALS','KITE SQUARE RECTANGLE RHOMBUS TRAPEZOID TRAPEZIUM PARALLELOGRAM DELTOID','Names for four-sided geometric figures')],
[('THROW','CAST HURL TOSS FLING CHUCK LOB PITCH SLING','Verbs for throwing an object'),('INDICATE','SPELL SIGNIFY MEAN PORTEND BETOKEN HERALD INDICATE SIGNAL','Verbs meaning foreshadow or stand for an outcome'),('JEWELRY ITEMS','CHARM RING BRACELET NECKLACE EARRING BROOCH LOCKET PENDANT','Objects worn as jewelry'),('SWEAR','CURSE CUSS SWEAR BLASPHEME DAMN RAIL RANT PROFANE','Verbs associated with uttering profanities or angry language')],
[('EXERCISE MOVEMENTS','PUMP PRESS CURL ROW SQUAT LUNGE CRUNCH PLANK','Named physical exercise actions'),('OBSTRUCT','CLOG BLOCK CHOKE PLUG STOPPER OCCLUDE OBSTRUCT JAM','Verbs meaning impede a passage'),('DISMISS','BOOT SACK FIRE AXE OUST EJECT EXPEL DISMISS','Verbs meaning remove someone from a role or place'),('IDLER','LOAFER SLACKER IDLER LAYABOUT SLUGGARD DAWDLER DRONE LOUNGER','Words for people avoiding work')]
]
stories=[
['A volunteer paused to CANVASS a leafy neighborhood.','A child nicknamed GOOBER offered to guide her.','His TOPKNOT caught a low branch.','They laughed beneath a CHESTNUT tree while admiring the woman\'s jewelry.'],
['An ITCHY traveler visited her COUSIN.','They put a small STAKE on who could prepare dinner faster.','She wore a formal DRESS beneath a borrowed apron.','Their talk drifted from clothes to a show on television.'],
['Two steep BANKS framed the river.','A tourist took a PIC from the bridge.','The boat stayed LEVEL as it approached the LAND.','At the dock, a vendor offered jewelry engraved with maps.'],
['A painted CREST hung above the concert stage.','A fan called DOJA pointed toward the lights.','SPARKS flew from a stage effect.','The audience in one QUADRANT cheered while a card trick unfolded nearby.'],
['A performer dressed as the GRINCH entered the rehearsal room.','A loose SPIKE on his costume snagged a music stand.','The bandleader was already a local LEGEND.','After playing through the score, they rested until the SABBATH and discussed new shoes.'],
['The boat crossed SHALLOW water.','Its MOTOR began to cough.','The driver said they COULD fix it with parts from his car.','His friend refused to play the FOOL and called a mechanic before their card game.'],
['A magician placed a SOLID box on the table.','He began to SHUFFLE a deck.','A painted MUSTACHE concealed his smile.','His assistant found the missing card among COMMON wildflowers beside some ornaments.'],
['A bare BULB lit the jeweler\'s window.','A silver BROOCH caught a shopper\'s eye.','Her BUDGET allowed one small purchase.','The TENDER exchange reminded her of a scene from a television show.'],
['A singer practiced the syllable SI.','MICHELLE joined the rehearsal.','They brushed a COBWEB from the television in the corner.','Their GLEE grew when the old set played a travel show about American cities.'],
['A YELLOW scarf lay on the table.','A SUNFLOWER stood beside it.','The WHITE walls made every shade vivid.','At the CENTER of the room, a gardener compared paint colors with tree leaves.'],
['A TABBY watched a fallen log begin to ROT.','Its owner waved a BATON to call a dog.','She offered to SHARE a wildlife magazine.','The photographs of large cats reminded her of patterns on a visiting musician\'s clothes.'],
['JOHN picked up a RUBY token.','A bowl of BEANS stood beside the board game.','His sister returned from SWIMMING to take her turn.','A documentary about magic played quietly while they rolled the dice.'],
['A bird chose a high PERCH.','In a JIFFY it rose above the river.','A WAVE broke below the cliff.','The flock seemed ready to ATTACK a passing kite as children sorted their shoes.'],
['A THIRST for adventure brought a young traveler to the fair.','A MAGICIAN invited her onstage.','His latest CREATION was a box with a hidden door.','She spotted its CONNECTION to an old shoe cabinet and laughed.'],
['A bowl of SOUP waited backstage.','The dancer practiced her FINAL turn.','A costume bag marked GUCCI hung nearby.','She pulled on her LEGGINGS and checked her shoes before the orchestra began.']]
# Local literal-fit corrections before export.
raw[7][1]=('___ JAM','PEARL TRAFFIC PAPER LOG TOE RASPBERRY STRAWBERRY APRICOT','Words preceding JAM in established expressions')
raw[9][0]=('___ RUSH','GOLD SUGAR HEAD LAND ADRENALINE LUNCH MORNING EVENING','Words preceding RUSH in established expressions')
raw[9][3]=('BEAT DECISIVELY','CREAM ROUT TROUNCE DRUB WALLOP CLOBBER POUND SMASH','Informal verbs for defeating an opponent badly')
raw[13][3]=('EXPRESS ANGER','CURSE CUSS SWEAR RAIL RANT RAVE FUME SEETHE','Verbs for expressing anger verbally or visibly')
scores=json.loads((D/'false_root_scores.json').read_text())['pools']
trees=[]
for i,(s,(cat,root),rs) in enumerate(zip(seed,roots,raw)):
    anchors=scores[s['id']]['selections']['yellow']['words']
    othercat,otherwords=roots[(i+7)%15]
    tree={'id':s['id'],'method':'false_connection','seed_words':s['seed_words'],'story':stories[i],
          'root_proposals':[{'category':cat,'words':root.split(),'connection':f'Each word is an example of {cat.lower()}.'},{'category':othercat,'words':otherwords.split(),'connection':f'Each word is an example of {othercat.lower()}.'}],
          'root_category':cat,'root_pool':root.split(),'false_group':{'category':cat,'words':anchors},'groups':[]}
    used=set()
    for j,((name,words,connection),anchor) in enumerate(zip(rs,anchors)):
        pool=words.split()
        assert len(pool)==8 and len(set(pool))==8,(s['id'],name,pool)
        assert anchor in pool,(s['id'],anchor,name)
        assert not (set(pool)&set(anchors)-{anchor}),(s['id'],name,'other anchor')
        assert not used&set(pool),(s['id'],name,used&set(pool))
        assert all(w.isalpha() for w in pool)
        used.update(pool)
        if '___' in name:
            fits={w:name.replace('___',w).replace('BOOK','BOOK').replace('HOLE','HOLE').replace('HOUSE','HOUSE').replace('BOAT','BOAT').replace('HEART','HEART') for w in pool}
            style='fill_in_the_blank'
        else:
            fits={w:f'{w}: {connection[0].lower()+connection[1:]}' for w in pool}
            style='synonyms_or_slang' if name in ['DIVERGE','COVER','UNADORNED','SEPARATION','CRITICIZE HARSHLY','THROW','EXHAUST','ABUNDANT','WEAR OUT','SUCCEED','FASCINATE','JOY','COMPANIONS','UNLOCATED','STRIKE','BEAT DECISIVELY','TAKE A CHANCE','BOTHER','AVOID','SELL','INDICATE','EXPRESS ANGER','OBSTRUCT','DISMISS','IDLER'] else 'names_of_things'
        tree['groups'].append({'id':f"{s['id']}-g{j}",'category':name,'style':style,'connection':connection,'words':pool,'required':[anchor],'anchor':anchor,'anchor_alternate_meaning':fits[anchor],'word_fits':fits})
    trees.append(tree)
(D/'false_trees.json').write_text(json.dumps(trees,indent=2)+'\n')
print('Created',len(trees),'false-connection families')

# Preserve the independent editor's targeted TAN repair on regeneration.
anagrams=dict(zip('TAN ACT GOD TAR LOW BARE FLOW SHORE'.split(),'ANT CAT DOG RAT OWL BEAR WOLF HORSE'.split()))
for tree in trees:
    for group in tree['groups']:
        if group['id']=='fc-10-g2':
            group.update(category='ANAGRAMMED ANIMALS',style='wordplay',connection='Rearrange the letters of each given word to obtain an animal name.',words=list(anagrams),anchor_alternate_meaning='TAN is an anagram of ANT; its letters supply the animal name rather than its color sense.',word_fits={w:f'{w} rearranges exactly to {a}.' for w,a in anagrams.items()})
(D/'false_trees.json').write_text(json.dumps(trees,indent=2)+'\n')
