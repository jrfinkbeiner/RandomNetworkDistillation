import sys
import pprint

from ase.db import connect



db_path_qm9 = "../qm9.db"
db_path_target = "../test_distribs/qm9/qm9_cossim_large-ext_run1_periter1_297cd0d0/targets.db"

with connect(db_path_target) as db:
    print(db.metadata)
    for row in db.select():
        print(row.toatoms())

from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList, RemoveOffsets, CastTo32

datapath = '../qm9.db'
qm9data = QM9(
    '../qm9.db', 
    batch_size=10,
    num_train=110000,
    num_val=10000,
    transforms=[
        ASENeighborList(cutoff=5.),
        # RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
        CastTo32()
    ],
)
qm9data.prepare_data()
qm9data.setup()


from schnetpack.data import AtomsLoader, ASEAtomsData

# target_database = ASEAtomsData(self.db_name)
target_dataset = ASEAtomsData(db_path_target, load_properties=["energy_U0", "target"], transforms=[
    ASENeighborList(cutoff=5.),
    # RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
    CastTo32()
],)

target_dataloader = AtomsLoader(
    target_dataset,
    batch_size=4,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
)

# target_database = ASEAtomsData(self.db_name)
qm9_dataset = qm9data.train_dataset #ASEAtomsData(db_path_qm9, load_properties=["energy_U0"])

# target_dataloader = AtomsLoader(
#     target_dataset,
#     batch_size=4,
#     num_workers=0,
#     shuffle=False,
#     pin_memory=True,
# )



for i in range(8):
    print(i, qm9_dataset[i]["energy_U0"], target_dataset[i]["energy_U0"])
    print(i, qm9_dataset[i]["_cell"].shape, target_dataset[i]["_cell"].shape)


print("\nqm9_dataset")
pprint.pprint(qm9_dataset[0].keys())
print("\ntarget_dataset")
pprint.pprint(target_dataset[0].keys())


# sys.exit()


print("\n--------------- Predictions ------------------")
rcut = 5.0 # LiCl: 5.5, NaCl: 6.0, KCl: 7.0
model_type = "large-ext"

from inp_helper import get_target_model
model = get_target_model(model_type, rcut)

for batch in target_dataloader:
    print(batch)
    # break
    result = model(batch)
    print("Result dictionary:", result)
    print("\n----------------------------------------------------")
    for key,item in result.items():
        print()
        print(key,item)
    break



# # dataloader_target = target_database.train_dataloader()

# for i,batch in enumerate(target_dataloader):
#     print()
#     print(i)
#     print(batch)
#     # if i>3