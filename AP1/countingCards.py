import random

# Dicionário de valores Hi-Lo para cada carta
card_values = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, 
    '7': 0, '8': 0, '9': 0, 
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

# Criando um baralho (pode ajustar para múltiplos baralhos)
deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4
random.shuffle(deck)

# Iniciando a contagem
running_count = 0
num_cards_to_show = 10  # Número de cartas para praticar

print("\n🃏 Bem-vindo ao Treino de Contagem de Cartas! 🃏\n")
input("Pressione ENTER para começar...\n")

# Mostra as cartas uma por uma
for i in range(num_cards_to_show):
    card = deck.pop()
    running_count += card_values[card]
    print(f"Carta {i+1}: {card}")
    input("Pressione ENTER para continuar...\n")

# Pergunta ao jogador qual foi a contagem
user_count = int(input("\n📊 Qual foi a contagem final? "))

# Verifica se está correto
if user_count == running_count:
    print("\n✅ Parabéns! Você contou corretamente!")
else:
    print(f"\n❌ Errou! A contagem correta era {running_count}.")

print("\n🎲 Jogue novamente para melhorar sua precisão!")