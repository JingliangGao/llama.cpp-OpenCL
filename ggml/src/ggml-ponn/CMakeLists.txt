if (DEFINED ENV{PONN_INSTALL_DIR})
    set(PONN_INSTALL_DIR $ENV{PONN_INSTALL_DIR})
endif()

if (NOT PONN_INSTALL_DIR)
    set(PONN_INSTALL_DIR "/usr/local/ponn")
endif()

if (EXISTS ${PONN_INSTALL_DIR})
    message(STATUS "PONN: PONN_INSTALL_DIR = ${PONN_INSTALL_DIR}")

    file(GLOB GGML_SOURCES_PONN "*.cpp")

    ggml_add_backend_library(ggml-ponn ${GGML_SOURCES_PONN})

    target_include_directories(ggml-ponn PRIVATE ${PONN_INSTALL_DIR}/include)
    target_link_directories(ggml-ponn PRIVATE ${PONN_INSTALL_DIR}/lib)

    find_library(PONN_LIB ponn PATHS ${PONN_INSTALL_DIR}/lib)
    if (PONN_LIB)
        target_link_libraries(ggml-ponn PRIVATE ${PONN_LIB})
    else()
        target_link_libraries(ggml-ponn PRIVATE ponn)
    endif()

    message(STATUS "PONN: enabled")
else()
    message(STATUS "PONN: PONN_INSTALL_DIR not found, disabling PONN backend")
    set(GGML_PONN OFF CACHE BOOL "Enable PONN backend" FORCE)
endif()