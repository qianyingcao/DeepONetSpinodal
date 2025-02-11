import numpy as np
import matplotlib.pyplot as plt

test_setups = list(range(12))

NUM_EPOCHS = 1000

NEGATIVE_NUMBER = -2.0

PLOT_ITS = list(range(100,1001,100))

epochs = np.linspace(0,NUM_EPOCHS,NUM_EPOCHS+1)

num_tests = len(test_setups)

loss_train_all = []
loss_test_all = []
L2_err_all = []
L2_relerr_all = []

for test in test_setups:
    case_name = 'testsetup_{}'.format(test)
    data_all = np.load(case_name + '/Output/SavedOutputs_final.npy', allow_pickle=True).item()
    loss_train_all.append(data_all['loss_train'])
    loss_test_all.append(data_all['loss_test'])
    L2_err_all.append(data_all['L2_err'])
    L2_relerr_all.append(data_all['L2_relerr'])

loss_train_all = np.array(loss_train_all)
loss_test_all = np.array(loss_test_all)
L2_err_all = np.array(L2_err_all)
L2_relerr_all = np.array(L2_relerr_all)

print(loss_train_all.shape)
print(np.array(L2_err_all).shape)
print(data_all['L2_err'].shape)

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (3.2*3,3.0*1))

axs[0].semilogy(epochs, loss_train_all.mean(axis=0), '-b', label='loss train (mean)')
axs[0].semilogy(epochs, loss_test_all.mean(axis=0), '-r', label='loss test (mean)')
axs[0].set_title('Loss')
axs[0].legend()

axs[1].errorbar(PLOT_ITS, L2_err_all.mean(axis=0), yerr=L2_err_all.std(axis=0), fmt='-o', color='b', label='L2 Err (mean)')
axs[2].errorbar(PLOT_ITS, L2_relerr_all.mean(axis=0), yerr=L2_relerr_all.std(axis=0), fmt='-o', color='r', label='L2 Rel Err (mean)')
#axs[1].plot(PLOT_ITS, L2_err_all.T, '.b', label='L2 Err (all)')
#axs[2].plot(PLOT_ITS, L2_relerr_all.T, '.r', label='L2 Rel Err (all)')
axs[1].set_yscale('log')
axs[2].set_yscale('log')
axs[1].set_title('Error')
axs[2].set_title('Relative Error')
#axs[1].legend()

fig.tight_layout()
plt.savefig('./post.png')
plt.close()

print('L2 Err mean: {:.4f}, std: {:.4f}'.format(L2_err_all[:,-1].mean(), L2_err_all[:,-1].std()))
print('L2 Rel Err mean: {:.4f}, std: {:.4f}'.format(L2_relerr_all[:,-1].mean(), L2_relerr_all[:,-1].std()))




fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (3.2*2,3.0*1))

axs[0].hist(L2_err_all[:,-1])
axs[1].hist(L2_relerr_all[:,-1])
fig.tight_layout()
plt.savefig('./err_hist.png')
plt.close()





dir_list = ['x','y','z']
num_subplots = 10

relerr_by_angle = {}

energy_all = []

def calc_energy(stress):
    loading_energy = np.nansum(stress[0])
    unloading_energy = np.nansum(stress[1])
    dissipation = 1 - unloading_energy/loading_energy
    return dissipation

for test in test_setups:
    print(test)
    case_name = 'testsetup_{}'.format(test)
    saved = np.load(case_name + '/Output/SavedOutputs_{}.npy'.format(NUM_EPOCHS), allow_pickle=True).item()
    stress_test_pred = saved['stress_test_pred']
    strain, stress_test, stress_weight_test, angle_test = np.load(case_name + '/Output/SavedOutputs_final.npy', allow_pickle=True).item()['data']
    #print(stress_test.shape)
    #print(angle_test.shape)
    fig, axs = plt.subplots(nrows = num_subplots, ncols = 3, figsize = (3.2*3,3.0*num_subplots))
    for icase in range(num_subplots):
        for idir in range(3):
            #print(icase)
            angle_case = angle_test[icase]
            angle_case_tuple = tuple(angle_case)
            energy_pred = calc_energy(stress_test_pred[icase,idir])
            energy_true = calc_energy(np.where(stress_test[icase,idir]==NEGATIVE_NUMBER, np.nan, stress_test[icase,idir]))
            energy_all.append([energy_pred, energy_true])
            for istage in range(stress_test_pred.shape[2]):
                axs[icase,idir].plot(strain, stress_test_pred[icase,idir,istage], '-b', label='pred')
                axs[icase,idir].plot(strain, np.where(stress_test[icase,idir,istage]==NEGATIVE_NUMBER, np.nan, stress_test[icase,idir,istage]), '-r', label='true')
            axs[icase,idir].legend()
            axs[icase,idir].set_title('{} {}'.format(dir_list[idir],angle_case))
            axs[icase,idir].set_xlim([-0.01,0.31])
            axs[icase,idir].set_ylim([-0.05,1.55])
            if angle_case_tuple in relerr_by_angle:
                relerr_by_angle[angle_case_tuple][1] += 1
                relerr_by_angle[angle_case_tuple][0] += np.sqrt(np.nanmean((stress_test_pred[icase,idir] - stress_test[icase,idir])**2))
            else:
                relerr_by_angle[angle_case_tuple] = [np.sqrt(np.nanmean((stress_test_pred[icase,idir] - stress_test[icase,idir])**2)), 1]
    fig.tight_layout()
    plt.savefig('./post_{}.png'.format(case_name))
    plt.close()

temp = []
for angle_tuple in relerr_by_angle:
    L2relerr_mean = relerr_by_angle[angle_tuple][0] / relerr_by_angle[angle_tuple][1]
    temp.append([angle_tuple, L2relerr_mean])

temp.sort(key=lambda x: x[1])
for angle, L2relerr in temp:
    print('L2 Rel. Err. {}: {:.4f}'.format(angle, L2relerr))



energy_all = np.array(energy_all)

plt.figure(111)
print(energy_all.shape)
plt.plot(energy_all[:,0],energy_all[:,1], '.b')
plt.xlabel('pred')
plt.ylabel('true')
fig.tight_layout()
plt.savefig('./energy_dissipation.png')
plt.close(111)

print(np.corrcoef(energy_all[:,0], energy_all[:,1]))