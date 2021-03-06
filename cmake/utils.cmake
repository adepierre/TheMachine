function(set_ide_paths_headers files)
    if(MSVC)
        foreach(source IN LISTS ${files})
            get_filename_component(source_path_header "${source}" PATH)
            string(REPLACE "include/${PROJECT_NAME}" "Headers/" source_path_header "${source_path_header}")
            string(REPLACE "/" "\\" source_path_msvc "${source_path_header}")
            source_group("${source_path_msvc}" FILES "${source}")
        endforeach()
    endif()
endfunction()

function(set_ide_paths_src files)
    if(MSVC)
        foreach(source IN LISTS ${files})
            get_filename_component(source_path "${source}" PATH)
            string(REPLACE "src" "Sources" source_path "${source_path}")
            string(REPLACE "/" "\\" source_path_msvc "${source_path}")
            source_group("${source_path_msvc}" FILES "${source}")
        endforeach()
    endif()
endfunction()
        

function(set_output target)
    if(MSVC)
        # To avoid having folder for each configuration when building with Visual
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/lib")
        set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/lib")
        set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/lib")
        set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}/lib")
    else()
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
        set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
    endif(MSVC)
endfunction()

function(set_debug_names target)
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${target} PROPERTIES RELWITHDEBINFO_POSTFIX "_rd")
endfunction()
