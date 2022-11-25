add_executable(get_started_with_IQS.exe get_started_with_IQS.cpp)
target_link_libraries(get_started_with_IQS.exe PRIVATE iqs)

add_executable(get_started_with_noisy_IQS.exe get_started_with_noisy_IQS.cpp)
target_link_libraries(get_started_with_noisy_IQS.exe iqs)

add_executable(bv_with_IQS.exe bv_with_IQS.cpp)
target_link_libraries(bv_with_IQS.exe PRIVATE iqs)


add_executable(ising_with_IQS.exe ising_with_IQS.cpp)
target_link_libraries(ising_with_IQS.exe PRIVATE iqs)

add_executable(grover_with_IQS.exe grover_with_IQS.cpp)
target_link_libraries(grover_with_IQS.exe PRIVATE iqs)

add_executable(qpe_with_IQS.exe qpe_with_IQS.cpp)
target_link_libraries(qpe_with_IQS.exe PRIVATE iqs)

add_executable(qnn_with_IQS.exe qnn_with_IQS.cpp)
target_link_libraries(qnn_with_IQS.exe PRIVATE iqs)

add_executable(qaoa_with_IQS.exe qaoa_with_IQS.cpp)
target_link_libraries(qaoa_with_IQS.exe PRIVATE iqs)

add_executable(cat_state_with_IQS.exe cat_state_with_IQS.cpp)
target_link_libraries(cat_state_with_IQS.exe PRIVATE iqs)

add_executable(cc_with_IQS.exe cc_with_IQS.cpp)
target_link_libraries(cc_with_IQS.exe PRIVATE iqs)

set_target_properties( get_started_with_IQS.exe
                       get_started_with_noisy_IQS.exe
                       bv_with_IQS.exe
                       ising_with_IQS.exe
                       grover_with_IQS.exe
                       qpe_with_IQS.exe
                       qnn_with_IQS.exe
                       qaoa_with_IQS.exe
                       cat_state_with_IQS.exe
                       cc_with_IQS.exe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
