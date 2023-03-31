"""algorithmic differentiation of thin film characteristic matrix"""

# Mjs = [] # list of characteristic matricies
# matprodbar = inputbar @ Aout.conj().T
# Mjs_back = reversed(Mjs)
# grads = []
# bar = matprodbar
# for i in range(len(Mjs_back)):
#     tmpbar = bar @ mjs_back[i].conj().T
#     extract_thickness_bar(tmpbar)
#     tensordot_thicknessbar_to_basis_funcs()
#     grads.append(*tensordot_return)
#     bar = tmpbar